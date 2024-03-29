import argparse

from utils.inference_model import model_fn, predict_fn

"""
This script runs inference using the finetuned model saved in pt file
Execution example:
python inference_pt.py \
    --model_dir models
    --abstract_text 'The proposed method re-frames traditional inverse problems of electrocardiography into regression problems, constraining the solution space by decomposing signals with multidimensional Gaussian impulse basis functions. Impulse HSPs were generated with single Gaussian basis functions at discrete heart surface locations and projected to corresponding BSPs using a volume conductor torso model. Both BSP (inputs) and HSP (outputs) were mapped to regular 2D surface meshes and used to train a neural network. Predictive capabilities of the network were tested with unseen synthetic and experimental data. A dense full connected single hidden layer neural network was trained to map body surface impulses to heart surface Gaussian basis functions for reconstructing HSP. Synthetic pulses moving across the heart surface were predicted from the neural network with root mean squared error of 9.1±1.4%. Predicted signals were robust to noise up to 20 dB and errors due to displacement and rotation of the heart within the torso were bounded and predictable. A shift of the heart 40 mm toward the spine resulted in a 4\% increase in signal feature localization error. The set of training impulse function data could be reduced and prediction error remained bounded. Recorded HSPs from in-vitro pig hearts were reliably decomposed using space-time Gaussian basis functions. Predicted HSPs for left-ventricular pacing had a mean absolute error of 10.4±11.4 ms. Other pacing scenarios were analyzed with similar success. Conclusion: Impulses from Gaussian basis functions are potentially an effective and robust way to train simple neural network data models for reconstructing HSPs from decomposed BSPs. The HSPs predicted by the neural network can be used to generate activation maps that non-invasively identify features of cardiac electrical dysfunction and can guide subsequent treatment options.'
"""

def main(args):
    model, tokenizer = model_fn(args.model_dir)
    predict_fn(args.abstract_text, model, tokenizer)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--abstract_text', type=str, default='The proposed method re-frames traditional inverse problems of electrocardiography into regression problems, constraining the solution space by decomposing signals with multidimensional Gaussian impulse basis functions.', help="the text of the abstract to classify")
    parser.add_argument('--model_dir', type=str, default='models', help="the directory where the model is saved as pt file")

    args = parser.parse_args()
    main(args)