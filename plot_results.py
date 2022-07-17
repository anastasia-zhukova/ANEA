from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import seaborn as sns

from machine_learning.argparser import parse_graph_arguments
from config import logger, EXEC_RES_PATH


def read_cvs(path: str):
    with open(path, "r") as file:
        return pd.read_csv(file)


def read_json(path: str):
    with open(path, "r") as file:
        return pd.read_json(file)


def create_barcharts(
    results: Dict[str, pd.DataFrame],
    datasets_used: Dict[str, List[str]],
    anea_date: str,
    topic: str,
) -> Path:
    """Creates two barcharts side by side based on the evaluation results from the classifier.

    :param Dict[str, pd.DataFrame] results: Transformed evaluation results from the classifier
    :param Dict[str, List[str]] datasets_used: Datasets that were used for the training of the classifier
    :param str anea_date: Date of the final_categories.json that has been used
    :param str topic: Topic of the final_categoreis.json
    :return Path save_path: Path to the figure on disk
    """
    # catplot to get two plots next to each other
    catplt = sns.catplot(
        x="HF_limit",
        y="score",
        hue="metric (LR)",
        col="avg",
        data=results["macro"],
        kind="bar",
        height=6,
        aspect=1.2,
    )
    catplt.fig.suptitle(f"{datasets_used}", fontsize="small", x=0.46, y=0.99)
    # get current time as YYYY-MM-dd_hh-mm-ss
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save figure to file
    save_path = Path(
        f"{EXEC_RES_PATH}/{anea_date}/{topic}/ML_classifier_results/comparison_{current_time}"
    )
    catplt.fig.savefig(save_path.with_suffix(".png"))
    return save_path


def transform_df(results: pd.DataFrame) -> pd.DataFrame:
    """Transforms the evaluation results from the classifier in a form that can be directly used by seaborns catplot.

    :param pd.DataFrame results: Evaluation results from the classifier
    :return pd.DataFrame: Transformed results required for plotting
    """
    metric_col = []
    score_col = []
    limit_col = []
    avg_col = []
    for row_id in range(len(results)):
        row = results.iloc[row_id]
        metric_col.append(f"test_accuracy ({row['learning_rate']})")
        score_col.append(row["test_accuracy"])
        limit_col.append(row["HF_limit"])
        avg_col.append(row["metric_average"])
        metric_col.append(f"balanced_acc ({row['learning_rate']})")
        score_col.append(row["balanced_acc"])
        limit_col.append(row["HF_limit"])
        avg_col.append(row["metric_average"])
        metric_col.append(f"F1 ({row['learning_rate']})")
        score_col.append(row["F1"])
        limit_col.append(row["HF_limit"])
        avg_col.append(row["metric_average"])
    return pd.DataFrame(
        {
            "metric (LR)": metric_col,
            "score": score_col,
            "HF_limit": limit_col,
            "avg": avg_col,
        }
    )


def run(args: Namespace):
    dataframes = []
    # read files
    for path in args.path:
        if Path(path).suffix == ".json":
            results = read_json(path)
            dataframes.append(results)
        elif Path(path).suffix == ".csv":
            results = read_cvs(path)
            dataframes.append(results)
        else:
            logger.error(f"Invalid suffix for: '{path}', skipping file!")
    # build single dataframe with all the data
    results = pd.concat(dataframes, ignore_index=True)
    # warn user if the files contain results for multiple datasets, dates or topics
    if len(results["out_of_domain_datasets"].unique()) > 1:
        logger.warning(
            f"Multiple different datasets used. Incorrect title! Datasets found: {results['out_of_domain_datasets']}"
        )
    if len(results["ANEA_date_used"].unique()) > 1:
        logger.warning(
            f"Multiple different dates used. {results['ANEA_date_used'].iloc[0]} will be used, but the comparison is invalid! Dates found: {results['ANEA_date_used']}"
        )
    if len(results["topic"].unique()) > 1:
        logger.warning(
            f"Multiple different topics used. {results['topic'].iloc[0]} will be used, but the comparison is invalid! Topics found: {results['topic']}"
        )
    # transform df for plotting
    transformed_results = transform_df(results)
    figure_path = create_barcharts(
        transformed_results,
        results["out_of_domain_datasets"].iloc[0],
        results["ANEA_date_used"].iloc[0],
        results["topic"].iloc[0],
        transformed_results["score"].max(),
    )
    logger.info(f"Figure has been saved to '{figure_path}'")


if __name__ == "__main__":
    args = parse_graph_arguments()
    run(args)
