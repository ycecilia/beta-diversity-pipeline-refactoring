import argparse


def parse_arguments():
    """
    Parses command-line arguments for a script.

    This function defines and parses command-line arguments required for fetching report data. Currently, it expects a single argument: the ID of the report to fetch. It uses the `argparse` module to facilitate command-line argument parsing, making it easy to extend for additional parameters in the future.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments. Specifically, it will have a `report_id` attribute containing the ID of the report as a string.

    Example:
        # Command-line usage
        # python script.py 12345

        args = parse_arguments()
        print(args.report_id)  # Outputs: 12345

    Note:
        This function is intended to be used in a command-line context where the report ID is passed as an argument to the script. Ensure that the report ID is provided when invoking the script, or the parser will display a help message and exit.
    """
    parser = argparse.ArgumentParser(
        description="Fetch report data based on report ID."
    )
    parser.add_argument("report_id", type=str, help="The ID of the report to fetch.")
    return parser.parse_args()
