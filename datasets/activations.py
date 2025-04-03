import torch
from torch.utils.data import Dataset
import os
import bisect

class ChunkedActivationsDataset(Dataset):
    def __init__(self, directory, transform=None, device="cpu"):
        """
        Args:
            directory (str): Path to the directory containing .pth files.
            transform (callable, optional): Optional transform to be applied on a sample.
            device (str): Device to store the activations ('cpu' or 'cuda').
        """
        self.directory = directory
        self.files = sorted(
            (f for f in os.listdir(directory) if f.endswith('.pth') or f.endswith('.pt')),
            key=lambda x: int(x.split('_part')[-1].split('.pt')[0])
        )
        self.transform = transform
        self.device = device
        self.file_offsets = []
        self.cumulative_lengths = []

        # Cache to store the last accessed file
        self._cached_file = None
        self._cached_tensor = None

        # Compute the offsets and cumulative lengths for efficient indexing
        cumulative = 0
        for file in self.files:
            tensor = torch.load(os.path.join(directory, file))  # Load file temporarily
            length = tensor.size(0)  # Number of samples in the file (K)
            self.file_offsets.append((file, length))
            cumulative += length
            self.cumulative_lengths.append(cumulative)
            # break

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _load_file(self, file):
        """
        Loads a file and caches it for future access.
        Moves the tensor to the specified device if not already cached.
        """
        if self._cached_file != file:
            file_path = os.path.join(self.directory, file)
            tensor = torch.load(file_path)
            self._cached_tensor = tensor.to(self.device)  # Move to the specified device
            self._cached_file = file

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            torch.Tensor: Neuron activation vector on the specified device.
        """
        # Find the correct file corresponding to the index
        for file_idx, (file, length) in enumerate(self.file_offsets):
            if idx < self.cumulative_lengths[file_idx]:
                # Calculate relative index within the file
                relative_idx = idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)

                # Load the file into cache if it's not already cached
                self._load_file(file)

                # Retrieve the sample
                sample = self._cached_tensor[relative_idx]
                if self.transform:
                    sample = self.transform(sample)
                return sample

        raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")


class ActivationsDataset(Dataset):
    # def __init__(self, directory, transform=None, device="cpu", take_every=1):
    #     """
    #     Args:
    #         directory (str): Path to the directory containing .pth files.
    #         transform (callable, optional): Optional transform to be applied on a sample.
    #         device (str): Device to store the activations ('cpu' or 'cuda').
    #     """
    #     self.directory = directory
    #     self.files = sorted(
    #         (f for f in os.listdir(directory) if f.endswith('.pth') or f.endswith('.pt')),
    #         key=lambda x: int(x.split('_part')[-1].split('.pt')[0])
    #     )
    #     self.transform = transform
    #     self.device = device
    #
    #     # Load all tensors into memory
    #     self.cached_tensors = []
    #     self.cumulative_lengths = []
    #     cumulative = 0
    #
    #     for file in self.files:
    #         tensor = torch.load(os.path.join(directory, file)).to(self.device)
    #         self.cached_tensors.append(tensor)
    #         cumulative += tensor.size(0)  # Number of samples in the file (K)
    #         self.cumulative_lengths.append(cumulative)
    def __init__(self, directory, transform=None, device="cpu", take_every=1):
        """
        Args:
            directory (str): Path to the directory containing .pth files.
            transform (callable, optional): Optional transform to be applied on a sample.
            device (str): Device to store the activations ('cpu' or 'cuda').
            take_every (int): Load every N-th row from the concatenated tensors to reduce memory usage.
        """
        self.directory = directory
        self.files = sorted(
            (f for f in os.listdir(directory) if (f.endswith('.pth') or f.endswith('.pt')) and not f.startswith('all')),
            key=lambda x: int(x.split('_part')[-1].split('.pt')[0])
        )
        self.transform = transform
        self.device = device
        self.take_every = take_every

        # Load all tensors into memory with skipping rows as per take_every
        self.cached_tensors = []
        self.cumulative_lengths = []
        cumulative = 0

        # Track the global row index
        global_row_index = 0

        for file in self.files:
            tensor = torch.load(os.path.join(directory, file)).to(self.device)
            num_rows = tensor.size(0)

            # Calculate global indices for the rows in this tensor
            global_indices = list(range(global_row_index, global_row_index + num_rows))
            selected_indices = [
                idx - global_row_index for idx in global_indices if idx % self.take_every == 0
            ]

            # Extract rows using the selected indices
            if selected_indices:
                tensor = tensor[selected_indices]
                self.cached_tensors.append(tensor)
                cumulative += len(selected_indices)
                self.cumulative_lengths.append(cumulative)

            # Update the global_row_index
            global_row_index += num_rows

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            torch.Tensor: Neuron activation vector on the specified device.
        """
        # # Find the correct tensor corresponding to the index
        # for tensor_idx, tensor in enumerate(self.cached_tensors):
        #     # return torch.randn_like(tensor[0]).to(self.device)
        #     if idx < self.cumulative_lengths[tensor_idx]:
        #         # Calculate relative index within the tensor
        #         relative_idx = idx - (self.cumulative_lengths[tensor_idx - 1] if tensor_idx > 0 else 0)
        #
        #         # Retrieve the sample
        #         sample = tensor[relative_idx]
        #         if self.transform:
        #             sample = self.transform(sample)
        #         return sample

        tensor_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        relative_idx = idx - (self.cumulative_lengths[tensor_idx - 1] if tensor_idx > 0 else 0)
        sample = self.cached_tensors[tensor_idx][relative_idx]
        if self.transform:
            sample = self.transform(sample)
        return sample