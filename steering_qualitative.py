from models.llava import Llava
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE
from PIL import Image
import requests

llava = Llava("cuda")
sae_path = "checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt"
sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()
text = "Write me a short love poem"
url = "https://img.freepik.com/free-photo/cement-texture_1194-5269.jpg?semt=ais_hybrid"
image = Image.open(requests.get(url, stream=True).raw)
for neuron in [39]: # Note: ID of the pencil neuron may change after retraining SAE
    for alpha in [0, 30, 40, 50]:
        print(f"neuron {neuron}, alpha {alpha}")
        llava.attach_and_fix(sae=sae, neurons_to_fix={neuron: alpha}, pre_zero=False)
        output = llava.prompt(text, image, max_tokens=60)[0]
        print(output)
        print("======================================================")