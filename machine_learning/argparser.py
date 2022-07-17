import argparse


def parse_test_arguments():
    """
    Parses the command line arguments for test parameters and selected datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        dest="topic",
        help="Topic of the ANEA run you want to use.",
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        dest="date",
        help="Date of the ANEA run you want to use (i.e. 2022-07-05 for 5th July 2022).",
    )
    parser.add_argument(
        "--batchsize",
        default=32,
        type=int,
        help="The size of batches used for training and evaluation of the model.",
    )
    parser.add_argument(
        "--epochs",
        default=[3],
        type=int,
        dest="num_epochs",
        help="Number of training epochs used for one test run.",
        nargs="*",
    )
    parser.add_argument(
        "--seed",
        default=[None],
        type=int,
        dest="random_seed",
        help="Random seed for the shuffling, for reproducability.",
        nargs="*",
    )
    parser.add_argument(
        "--testsize",
        default=0.2,
        type=float,
        dest="testset_size",
        help="Size of the test dataset.",
    )
    parser.add_argument(
        "--trainsize",
        default=0.8,
        type=float,
        dest="trainset_size",
        help="Size of the training dataset.",
    )
    parser.add_argument(
        "--trials",
        default=1,
        type=int,
        dest="trials",
        help="How many models should be trained. Will only use the best one for evaluation",
    )
    parser.add_argument(
        "--avg",
        default=["macro"],
        type=str,
        choices=["macro", "none", "binary", "weighted", "samples"],
        dest="average",
        help="Parameter required for multiclass targets. Defaults to 'macro'. Determines how the evaluation scores are calculated, for further information see i.e. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score",
        nargs="*",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=[1e-5],
        type=float,
        dest="learning_rate",
        help="Learning rate for model training.",
        nargs="*",
    )
    parser.add_argument(
        "--step_size",
        default=[1],
        type=int,
        dest="step_size",
        help="Period of learning rate decay (in epochs)",
        nargs="*",
    )
    parser.add_argument(
        "--label_dist",
        action="store_true",
        dest="label_dist",
        help="Whether to use the distribution of labels as weights for the cross entropy loss",
    )
    parser.add_argument(
        "--fine_tuning_steps",
        default=[4],
        type=int,
        dest="ft_steps",
        help="Number of fine tuning iterations for each model while reducing the out-of-domain data.",
        nargs="*",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="If this flag is set the results will be saved as a JSON instead of a CSV file.",
    )
    parser.add_argument(
        "--hf_datasets",
        default=[],
        type=str,
        dest="hf_datasets",
        choices=["News", "Amazon", "Books"],
        help="Choose one or multiple of the predefined HuggingFace datasets to use as general out of domain training data. Each choosen set will be limited to the amount set by --hf_limit.",
        nargs="*",
    )
    parser.add_argument(
        "--hf_limit",
        default=None,
        type=int,
        dest="hf_limit",
        help="Maximum number of samples that will be used from each selected HuggingFace dataset.",
    )
    parser.add_argument(
        "--own_hf_dataset",
        default=[],
        type=str,
        dest="own_hf_dataset",
        help="Used to select any dataset from HuggingFace Datasets and use it as additional out of domain data. Usage: --own_hf_dataset path name split colname_data lang limit shuffle",
        nargs="*"
    )
    parser.add_argument(
        "--anea_datasets",
        default=[],
        type=str,
        dest="anea_datasets",
        help="Use any amount of datasets (topics) that were created by running ANEA as additional general out of domain data.",
        nargs="*",
    )
    return parser.parse_args()


def parse_graph_arguments():
    """
    Parses the command line arguments for the graph creation module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=[],
        type=str,
        dest="path",
        help="One or multiple paths to result files.",
        nargs="+",
    )
    return parser.parse_args()
