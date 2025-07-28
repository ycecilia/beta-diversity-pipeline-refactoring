"""
Simplified labels module for challenge - contains only the functions needed by alpha.py.
"""
import pandas as pd
import polars as pl


def extract_legend_and_axis_label(labels_and_legend, environmental_variable):
    """
    Extract the legend and x-axis label for a given environmental variable from a DataFrame.

    Args:
    labels_and_legend (pandas.DataFrame or polars.DataFrame): DataFrame containing the labels and legends.
    environmental_variable (str): The environmental variable to filter the DataFrame on.

    Returns:
    tuple: A tuple containing the legend and x-axis label for the given environmental variable.
           Returns (None, None) if the environmental variable is not found.
    """
    # Handle both pandas and polars DataFrames
    if isinstance(labels_and_legend, pl.DataFrame):
        filtered_df = labels_and_legend.filter(
            pl.col("Environmental_Variable") == environmental_variable
        )
        
        if filtered_df.height == 0:
            return None, None
        
        label_val = filtered_df.select("Label").item(0, 0)
        axis_label_val = filtered_df.select("Axis_Label").item(0, 0)
        
        legend = label_val if label_val is not None and str(label_val) != "null" else None
        axis_label = axis_label_val if axis_label_val is not None and str(axis_label_val) != "null" else None
        
    else:  # pandas DataFrame
        filtered_df = labels_and_legend[
            labels_and_legend["Environmental_Variable"] == environmental_variable
        ]

        if filtered_df.empty:
            return None, None

        legend = filtered_df["Label"].values[0] if not pd.isna(filtered_df["Label"].values[0]) else None
        axis_label = filtered_df["Axis_Label"].values[0] if not pd.isna(filtered_df["Axis_Label"].values[0]) else None

    return legend, axis_label


def adjust_metadata_with_categories(metadata, categories, environmental_variable):
    """
    Adjust metadata with categories for a given environmental variable.
    
    Args:
    metadata (pandas.DataFrame): Metadata DataFrame
    categories (pandas.DataFrame or polars.DataFrame): Categories DataFrame
    environmental_variable (str): Environmental variable name
    
    Returns:
    pandas.DataFrame: Adjusted metadata
    """
    if environmental_variable == "iucn_cat":
        metadata[environmental_variable] = metadata[environmental_variable].fillna("not reported")

    # Handle both pandas and polars DataFrames for categories
    if isinstance(categories, pl.DataFrame):
        env_vars_list = categories["Environmental_Variable"].to_list()
    else:
        env_vars_list = categories["Environmental_Variable"].tolist()

    # Adjust metadata with categories
    if (
        environmental_variable in metadata.columns
        and environmental_variable in env_vars_list
    ):
        # Convert polars to pandas if needed for processing
        if isinstance(categories, pl.DataFrame):
            categories_pd = categories.to_pandas()
        else:
            categories_pd = categories

        # Filter categories for the specific environmental variable
        relevant_categories = categories_pd[
            categories_pd["Environmental_Variable"] == environmental_variable
        ]

        # Rename the environmental_variable column in metadata to 'value'
        metadata = metadata.rename(columns={environmental_variable: "value"})

        # Merge metadata with categories on 'value'
        metadata = metadata.merge(relevant_categories, on="value", how="left")

        # Rename the 'description' column back to the original environmental_variable name
        if "description" in metadata.columns:
            metadata = metadata.rename(columns={"description": environmental_variable})

        print("Finished changing values of selected variable.")
    elif environmental_variable in metadata.columns:
        metadata["value"] = metadata[environmental_variable]
    else:
        print(f"{environmental_variable} is not a column in the metadata DataFrame.")

    return metadata