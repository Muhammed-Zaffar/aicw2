import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def RemoveNaN(df):
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors='coerce')
	df = df.apply(pd.to_numeric, errors='coerce')

	# Replace -999 values with NaN
	df = df.replace(-999, np.nan)
	clean_df = df.dropna()
	clean_df.reset_index(drop=True, inplace=True)
	return clean_df


def removeAnomalies(clean_df):
	# Apply the scaling function to each column in the DataFrame
	mean = clean_df.mean()
	std = clean_df.std()
	z_scores = (clean_df - mean) / std

	# Keep only rows where all column z-scores are less than 3 in absolute value
	clean_df = clean_df[(np.abs(z_scores) < 3).all(axis=1)]
	clean_df.reset_index(drop=True, inplace=True)
	return clean_df


def scale_column(column, min_scale=0.9, max_scale=0.1):
	min_value = column.min()
	max_value = column.max()
	scaled_column = min_scale + ((column - min_value) * (max_scale - min_scale)) / (max_value - min_value)
	return scaled_column, min_value, max_value


def standardize_df(df, mins, maxes, lower=0.1, upper=0.9):
	# Standardizes all values in a DataFrame to a specific range [lower, upper]
	# using provided min and max values.
	std_df = lower + ((df - mins) * (upper - lower) / (maxes - mins))
	return std_df


def destandardize_df(std_df, min_val, max_val, lower=0.1, upper=0.9):
	# De-standardizes a DataFrame from a specific range [lower, upper] to its original scale
	# using provided min and max values.
	orig_df = min_val + ((std_df - lower) * (max_val - min_val) / (upper - lower))
	return orig_df


if __name__ == "__main__":
	# Load the data
	df = pd.read_csv(
		r"C:\Users\kaduj\OneDrive - Loughborough University\Year 2\23COB107 - AI Methods\Sem 2\coursework\FEHDataStudent_csv.csv",
		delimiter=',', dtype={'SAAR': 'Int64'})

	print(f"{df=}")

	clean_df = RemoveNaN(df)
	print(f"{clean_df=}")

	clean_df = removeAnomalies(clean_df)
	print(f"{clean_df=}")

	# Calculate indices for splits
	train_end = int(len(clean_df) * 0.6)
	validate_end = int(len(clean_df) * 0.8)

	# Split the DataFrame
	train_df = clean_df.iloc[:train_end]
	validate_df = clean_df.iloc[train_end:validate_end]
	test_df = clean_df.iloc[validate_end:]

	# Min-Max scaling training and validation data
	# train_df = train_df.apply(lambda x: scale_column(x))
	# validate_df = validate_df.apply(lambda x: scale_column(x))
	# test_df = test_df.apply(lambda x: scale_column(x))

	combined_df = pd.concat([train_df, validate_df])
	mins = combined_df.min()
	maxes = combined_df.max()

	train_df = standardize_df(train_df, mins, maxes)
	validate_df = standardize_df(validate_df, mins, maxes)
	test_df = standardize_df(test_df, mins, maxes)

	# output the dataframes to csv
	train_df.to_csv("train.csv", index=False)
	validate_df.to_csv("validate.csv", index=False)
	test_df.to_csv("test.csv", index=False)

	print(train_df)
	print("-------------------")
	print(validate_df)
	print("-------------------")
	print(test_df)
