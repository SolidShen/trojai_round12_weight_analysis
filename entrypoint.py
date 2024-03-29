""" Entrypoint to interact with the detector.
"""
import json
import logging
import warnings

import jsonschema

from detector import WeightAnalysisDetector

warnings.filterwarnings("ignore")

import random 

import os 
import torch 
import numpy as np 


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def inference_mode(args):
    
    
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)


    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = WeightAnalysisDetector(args.metaparameters_filepath, args.learned_parameters_dirpath, args.scale_parameters_filepath)

    logging.info("Calling the trojan detector")
    detector.infer(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, args.round_training_dataset_dirpath)
def configure_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    
    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = WeightAnalysisDetector(args.metaparameters_filepath, args.learned_parameters_dirpath, args.scale_parameters_filepath)

    logging.info("Calling configuration mode")
    detector.configure(args.configure_models_dirpath, args.automatic_configuration)

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    # seed_torch(42)

    temp_parser = ArgumentParser(add_help=False)

    parser = ArgumentParser(
        description="Template Trojan Detector to Demonstrate Test and Evaluation. Should be customized to work with target round in TrojAI."
        "Infrastructure."
    )

    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    inf_parser = subparser.add_parser('infer', help='Execute container in inference mode for TrojAI detection.')

    inf_parser.add_argument(
        "--model_filepath",
        type=str,
        help="File path to the pytorch model file to be evaluated.",
        required=True
    )
    inf_parser.add_argument(
        "--result_filepath",
        type=str,
        help="File path to the file where output result should be written. After "
        "execution this file should contain a single line with a single floating "
        "point trojan probability.",
        required=True
    )
    inf_parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        required=True
    )
    inf_parser.add_argument(
        "--examples_dirpath",
        type=str,
        help="File path to the folder of examples which might be useful for determining "
        "whether a model is poisoned.",
        required=True
    )
    inf_parser.add_argument(
        "--round_training_dataset_dirpath",
        type=str,
        help="File path to the directory containing id-xxxxxxxx models of the current "
        "rounds training dataset.",
        required=True
    )

    inf_parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )
    inf_parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )
    inf_parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )
    inf_parser.add_argument(
        "--scale_parameters_filepath",
        type=str,
        help="Path to a .npy file containing the mean and scale which are used to "
        "normalize the feature vectors before inferencing.",
        required=True,
    )


    inf_parser.set_defaults(func=inference_mode)


    configure_parser = subparser.add_parser('configure', help='Execute container in configuration mode for TrojAI detection. This will produce a new set of learned parameters to be used in inference mode.')

    configure_parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        required=True
    )

    configure_parser.add_argument(
        "--configure_models_dirpath",
        type=str,
        help="Path to a directory containing models to use when in configure mode.",
        required=True,
    )

    configure_parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )

    configure_parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )

    configure_parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )
    configure_parser.add_argument(
        '--automatic_configuration',
        help='Whether to enable automatic training or not, which will retrain the detector across multiple variables',
        action='store_true',
    )
    configure_parser.add_argument(
        "--scale_parameters_filepath",
        type=str,
        help="Path to a .npy file containing the mean and scale which are used to "
        "normalize the feature vectors before inferencing.",
        required=True,
    )

    configure_parser.set_defaults(func=configure_mode)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

    args, extras = temp_parser.parse_known_args()

    if '--help' in extras or '-h' in extras:
        args = parser.parse_args()
    # Checks if new mode of operation is being used, or is this legacy
    elif len(extras) > 0 and extras[0] in ['infer', 'configure']:
        args = parser.parse_args()
        args.func(args)

    else:
        # Assumes we have inference mode if the subparser is not used
        args = inf_parser.parse_args()
        args.func(args)
