"""
Generate all features

Part of the code has been copied from Spark RAPIDS examples github repo
https://github.com/NVIDIA/spark-rapids-examples/tree/main

"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.sql import functions as F

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType

from tsfresh.feature_extraction import feature_calculators

import numpy as np
import pandas as pd
import scipy as sp

from xgboost.spark import SparkXGBRegressor, SparkXGBRegressorModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from datetime import datetime
from time import time


### Functions
def with_benchmark(phrase, action):
    start = time()
    result = action()
    end = time()
    print('{} takes {} seconds'.format(phrase, round(end - start, 2)))
    return result


def transform():
    result = model.transform(df_test).cache()
    result.foreachPartition(lambda _: None)
    return result


def generate_features(df_fg):
    df_fg = df_fg.withColumn("mean", F.udf(lambda x: float(np.mean(x)))("ad").cast("Float")) \
        .withColumn("max", F.udf(lambda x: float(np.max(x)))("ad").cast("Float")) \
        .withColumn("min", F.udf(lambda x: float(np.min(x)))("ad").cast("Float")) \
        .withColumn("std", F.udf(lambda x: float(np.std(x)))("ad").cast("Float")) \
        .withColumn("var", F.udf(lambda x: float(np.var(x)))("ad").cast("Float")) \
        .withColumn("ptp", F.udf(lambda x: float(np.ptp(x)))("ad").cast("Float")) \
        .withColumn("percentile_10", F.udf(lambda x: float(np.percentile(x, 10)))("ad").cast("Float")) \
        .withColumn("percentile_20", F.udf(lambda x: float(np.percentile(x, 20)))("ad").cast("Float")) \
        .withColumn("percentile_30", F.udf(lambda x: float(np.percentile(x, 30)))("ad").cast("Float")) \
        .withColumn("percentile_40", F.udf(lambda x: float(np.percentile(x, 40)))("ad").cast("Float")) \
        .withColumn("percentile_50", F.udf(lambda x: float(np.percentile(x, 50)))("ad").cast("Float")) \
        .withColumn("percentile_60", F.udf(lambda x: float(np.percentile(x, 60)))("ad").cast("Float")) \
        .withColumn("percentile_70", F.udf(lambda x: float(np.percentile(x, 70)))("ad").cast("Float")) \
        .withColumn("percentile_80", F.udf(lambda x: float(np.percentile(x, 80)))("ad").cast("Float")) \
        .withColumn("percentile_90", F.udf(lambda x: float(np.percentile(x, 90)))("ad").cast("Float")) \
 \
        .withColumn("skew", F.udf(lambda x: float(sp.stats.skew(x)))("ad").cast("Float")) \
        .withColumn("kurtosis", F.udf(lambda x: float(sp.stats.kurtosis(x)))("ad").cast("Float")) \
        .withColumn("kstat_1", F.udf(lambda x: float(sp.stats.kstat(x, 1)))("ad").cast("Float")) \
        .withColumn("kstat_2", F.udf(lambda x: float(sp.stats.kstat(x, 2)))("ad").cast("Float")) \
        .withColumn("kstat_3", F.udf(lambda x: float(sp.stats.kstat(x, 3)))("ad").cast("Float")) \
        .withColumn("kstat_4", F.udf(lambda x: float(sp.stats.kstat(x, 4)))("ad").cast("Float")) \
        .withColumn("moment_1", F.udf(lambda x: float(sp.stats.moment(x, 1)))("ad").cast("Float")) \
        .withColumn("moment_2", F.udf(lambda x: float(sp.stats.moment(x, 2)))("ad").cast("Float")) \
        .withColumn("moment_3", F.udf(lambda x: float(sp.stats.moment(x, 3)))("ad").cast("Float")) \
        .withColumn("moment_4", F.udf(lambda x: float(sp.stats.moment(x, 4)))("ad").cast("Float")) \
 \
        .withColumn("abs_energy", F.udf(lambda x: float(feature_calculators.abs_energy(x)))("ad").cast("Float")) \
        .withColumn("abs_sum_of_changes", F.udf(lambda x: float(feature_calculators.absolute_sum_of_changes(x)))("ad").cast("Float")) \
        .withColumn("count_above_mean", F.udf(lambda x: float(feature_calculators.count_above_mean(x)))("ad").cast("Float")) \
        .withColumn("count_below_mean", F.udf(lambda x: float(feature_calculators.count_below_mean(x)))("ad").cast("Float")) \
        .withColumn("mean_abs_change", F.udf(lambda x: float(feature_calculators.mean_abs_change(x)))("ad").cast("Float")) \
        .withColumn("mean_change", F.udf(lambda x: float(feature_calculators.mean_change(x)))("ad").cast("Float")) \
        .withColumn("var_larger_than_std_dev",
                    F.udf(lambda x: float(feature_calculators.variance_larger_than_standard_deviation(x)))("ad").cast("Float")) \
 \
        .withColumn("range_minf_m4000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -np.inf, -4000)))("ad").cast("Float")) \
        .withColumn("range_m4000_m3000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -4000, -3000)))("ad").cast("Float")) \
        .withColumn("range_m3000_m2000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -3000, -2000)))("ad").cast("Float")) \
        .withColumn("range_m2000_m1000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -2000, -1000)))("ad").cast("Float")) \
        .withColumn("range_m1000_0",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -1000, 0)))("ad").cast("Float")) \
        .withColumn("range_0_p1000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 0, 1000)))("ad").cast("Float")) \
        .withColumn("range_p1000_p2000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 1000, 2000)))("ad").cast("Float")) \
        .withColumn("range_p2000_p3000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 2000, 3000)))("ad").cast("Float")) \
        .withColumn("range_p3000_p4000",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 3000, 4000)))("ad").cast("Float")) \
        .withColumn("range_p4000_pinf",
                    F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 4000, np.inf)))("ad").cast("Float")) \
 \
        .withColumn("ratio_unique_values",
                    F.udf(lambda x: float(feature_calculators.ratio_value_number_to_time_series_length(x)))("ad").cast("Float")) \
        .withColumn("first_loc_min", F.udf(lambda x: float(feature_calculators.first_location_of_minimum(x)))("ad").cast("Float")) \
        .withColumn("first_loc_max", F.udf(lambda x: float(feature_calculators.first_location_of_maximum(x)))("ad").cast("Float")) \
        .withColumn("last_loc_min", F.udf(lambda x: float(feature_calculators.last_location_of_minimum(x)))("ad").cast("Float")) \
        .withColumn("last_loc_max", F.udf(lambda x: float(feature_calculators.last_location_of_maximum(x)))("ad").cast("Float")) \
 \
        .withColumn("time_rev_asym_stat_10",
                    F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 10)))("ad").cast("Float")) \
        .withColumn("time_rev_asym_stat_100",
                    F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 100)))("ad").cast("Float")) \
        .withColumn("time_rev_asym_stat_1000",
                    F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 1000)))("ad").cast("Float")) \
 \
        .withColumn("autocorrelation_5", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 5)))("ad").cast("Float")) \
        .withColumn("autocorrelation_10", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 10)))("ad").cast("Float")) \
        .withColumn("autocorrelation_50", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 50)))("ad").cast("Float")) \
        .withColumn("autocorrelation_100", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 100)))("ad").cast("Float")) \
        .withColumn("autocorrelation_1000", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 1000)))("ad").cast("Float")) \
 \
        .withColumn("c3_5", F.udf(lambda x: float(feature_calculators.c3(x, 5)))("ad").cast("Float")) \
        .withColumn("c3_10", F.udf(lambda x: float(feature_calculators.c3(x, 10)))("ad").cast("Float")) \
        .withColumn("c3_100", F.udf(lambda x: float(feature_calculators.c3(x, 100)))("ad").cast("Float")) \
 \
        .withColumn("fft_1_real", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'real'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_1_imag", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'imag'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_1_angle", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'angle'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_2_real", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'real'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_2_imag", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'imag'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_2_angle", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'angle'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_3_real", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'real'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_3_imag", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'imag'}]))[0][1]))("ad").cast("Float")) \
        .withColumn("fft_3_angle", F.udf(
        lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'angle'}]))[0][1]))("ad").cast("Float")) \
 \
        .withColumn("long_strk_above_mean",
                    F.udf(lambda x: float(feature_calculators.longest_strike_above_mean(x)))("ad").cast("Float")) \
        .withColumn("long_strk_below_mean",
                    F.udf(lambda x: float(feature_calculators.longest_strike_below_mean(x)))("ad").cast("Float")) \
        .withColumn("cid_ce_0", F.udf(lambda x: float(feature_calculators.cid_ce(x, 0)))("ad").cast("Float")) \
        .withColumn("cid_ce_1", F.udf(lambda x: float(feature_calculators.cid_ce(x, 1)))("ad").cast("Float")) \
 \
        .withColumn("binned_entropy_5", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 5)))("ad").cast("Float")) \
        .withColumn("binned_entropy_10", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 10)))("ad").cast("Float")) \
        .withColumn("binned_entropy_20", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 20)))("ad").cast("Float")) \
        .withColumn("binned_entropy_50", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 50)))("ad").cast("Float")) \
        .withColumn("binned_entropy_80", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 80)))("ad").cast("Float")) \
        .withColumn("binned_entropy_100", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 100)))("ad").cast("Float")) \
 \
        .withColumn("num_crossing_0", F.udf(lambda x: float(feature_calculators.number_crossing_m(x, 0)))("ad").cast("Float")) \
        .withColumn("num_peaks_10", F.udf(lambda x: float(feature_calculators.number_peaks(x, 10)))("ad").cast("Float")) \
        .withColumn("num_peaks_50", F.udf(lambda x: float(feature_calculators.number_peaks(x, 50)))("ad").cast("Float")) \
        .withColumn("num_peaks_100", F.udf(lambda x: float(feature_calculators.number_peaks(x, 100)))("ad").cast("Float")) \
        .withColumn("num_peaks_500", F.udf(lambda x: float(feature_calculators.number_peaks(x, 500)))("ad").cast("Float")) \
 \
        .withColumn("spkt_welch_density_1",
                    F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]))("ad").cast("Float")) \
        .withColumn("spkt_welch_density_10",
                    F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]))("ad").cast("Float")) \
        .withColumn("spkt_welch_density_50",
                    F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]))("ad").cast("Float")) \
        .withColumn("spkt_welch_density_100",
                    F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]))("ad").cast("Float"))
    return df_fg


spark = SparkSession \
    .builder \
    .appName("Final_code") \
    .getOrCreate()

spark

start = time()
schema = StructType([
    StructField('acoustic_data', FloatType(), True),
    StructField('time_to_failure', DoubleType(), True)
])

df = spark.read.csv('s3://subhoms-test/input/data/LANL/LANL_Earthquake_Prediction/train.csv', header=True,
                    schema=schema)
df.show(10)

group_expr = 'MONOTONICALLY_INCREASING_ID() DIV 150000'
df = df.withColumn('idx', F.expr(group_expr))

df = df.repartition(6).cache()
print(df.rdd.getNumPartitions())

df_features = df.groupby('idx').agg(F.collect_list('acoustic_data').alias("ad"), F.last('time_to_failure').alias("ttf"))
df_features.show(10, truncate=False)

df_features = generate_features(df_features)

end = time()
print('Feature generation takes {} seconds'.format(round(end - start, 2)))

df_features = df_features.drop('idx').drop('ad')
#df_features.write.option("header", True).csv("s3://subhoms-test/input/data/LANL/LANL_Earthquake_Prediction/sparkfeatures")

label = 'ttf'
features = [ x for x in df_features.columns if x != label ]

# Create XGBoostRegressor
params = {
    "tree_method": "gpu_hist",
    "grow_policy": "depthwise",
    "num_workers": 2,
    "use_gpu": "true",
}
params['features_col'] = features
params['label_col'] = label

regressor = SparkXGBRegressor(**params)

# Then build the evaluator and the hyperparameters
evaluator = (RegressionEvaluator()
             .setLabelCol(label))
param_grid = (ParamGridBuilder()
              .addGrid(regressor.max_depth, [3, 6])
              .addGrid(regressor.n_estimators, [100, 200])
              .build())
# Finally the corss validator
cross_validator = (CrossValidator()
                   .setEstimator(regressor)
                   .setEvaluator(evaluator)
                   .setEstimatorParamMaps(param_grid)
                   .setNumFolds(2))

# Train
model = with_benchmark('Cross-Validation', lambda: cross_validator.fit(df_features)).bestModel

test_schema = StructType([
    StructField('acoustic_data', FloatType(), True)
])

df_test = spark.read.csv('s3://subhoms-test/input/data/LANL/LANL_Earthquake_Prediction/test/seg_00030f.csv',
                         header=True,
                         schema=test_schema)

df_test = df_test.withColumn('idx', F.lit(1)) \
    .groupby('idx').agg(F.collect_list('acoustic_data')) \
    .withColumnRenamed('collect_list(acoustic_data)', 'ad')

df_test = generate_features(df_test)
df_test = df_test.drop('idx').drop('ad')

# Transform
result = with_benchmark('Transforming', transform)
result.select(label, 'prediction').show(5)

# Evaluate
accuracy = with_benchmark(
    'Evaluation',
    lambda: RegressionEvaluator().setLabelCol(label).evaluate(result))
print('RMSE is ' + str(accuracy))

# Stop the Spark session
spark.stop()
