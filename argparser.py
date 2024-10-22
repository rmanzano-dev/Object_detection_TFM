import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Image Prediction Script")

    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        help="Pretrained model"
    )
    # Adding a flag for enabling predict mode
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'val'],  # The allowed options
        help="Select the mode: 'train', 'predict', or 'val'"
    )


    # Adding an argument for the image path
    parser.add_argument(
        '--img_path', 
        type=str,
        help="Path to the input image"
    )

    parser.add_argument(
        '--output_path', 
        type=str,
        help="Path to the input image"
    )

    parser.add_argument(
        '--input_path', 
        type=str,
        help="Path to the input image"
    )

    parser.add_argument(
        '--epochs', 
        type=int,
        help="Number of epochs"
    )

    # Adding a flag for enabling predict mode
    parser.add_argument(
        '--exp_name',
        type=str,
        help="Name of the experiment to save checkpoint"
    )


    args = parser.parse_args()
    return args
