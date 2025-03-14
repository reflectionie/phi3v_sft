import argparse
from transformers import AutoModelForCausalLM

def save_model(save_path):
    """
    Load the Phi-3.5-vision-instruct model and save it to the specified path.

    Args:
        save_path (str): The directory to save the model.
    """
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
    print("Saving the model...")
    model.save_pretrained(save_path, safe_serialization=False)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the Phi-3.5-vision-instruct model to a specified path.")
    parser.add_argument("save_path", type=str, help="The target directory to save the model.")
    args = parser.parse_args()

    save_model(args.save_path)
