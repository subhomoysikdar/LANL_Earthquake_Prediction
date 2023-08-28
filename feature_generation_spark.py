"""
Generate all features
"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

from pyspark.sql import functions as F

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType

from tsfresh.feature_extraction import feature_calculators

import numpy as np
import pandas as pd
import scipy as sp

from datetime import datetime
from time import time

spark = SparkSession \
    .builder \
    .appName("All_Feature_Generation") \
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

df=df.repartition(6).cache()
print(df.rdd.getNumPartitions())

df_features = df.groupby('idx').agg(F.collect_list('acoustic_data').alias("ad"), F.last('time_to_failure').alias("ttf"))
df_features.show(10, truncate=False)

rdd_features = df_features.rdd.map(lambda x:
                                   (
                                       x[0],
                                       x[2],
                                       float(np.mean(x[1])),
                                       float(np.max(x[1])),
                                       float(np.min(x[1])),
                                       float(np.std(x[1])),
                                       float(np.var(x[1])),
                                       float(np.ptp(x[1])),
                                       float(np.percentile(x[1], 10)),
                                       float(np.percentile(x[1], 20)),
                                       float(np.percentile(x[1], 30)),
                                       float(np.percentile(x[1], 40)),
                                       float(np.percentile(x[1], 50)),
                                       float(np.percentile(x[1], 60)),
                                       float(np.percentile(x[1], 70)),
                                       float(np.percentile(x[1], 80)),
                                       float(np.percentile(x[1], 90)),

                                       float(sp.stats.skew(x[1])),
                                       float(sp.stats.kurtosis(x[1])),
                                       float(sp.stats.kstat(x[1], 1)),
                                       float(sp.stats.kstat(x[1], 2)),
                                       float(sp.stats.kstat(x[1], 3)),
                                       float(sp.stats.kstat(x[1], 4)),
                                       float(sp.stats.moment(x[1], 1)),
                                       float(sp.stats.moment(x[1], 2)),
                                       float(sp.stats.moment(x[1], 3)),
                                       float(sp.stats.moment(x[1], 4)),

                                       float(feature_calculators.abs_energy(x[1])),
                                       float(feature_calculators.absolute_sum_of_changes(x[1])),
                                       float(feature_calculators.count_above_mean(x[1])),
                                       float(feature_calculators.count_below_mean(x[1])),
                                       float(feature_calculators.mean_abs_change(x[1])),
                                       float(feature_calculators.mean_change(x[1])),
                                       float(feature_calculators.variance_larger_than_standard_deviation(x[1])),

                                       float(feature_calculators.range_count(np.asarray(x[1]), -np.inf, -4000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), -4000, -3000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), -3000, -2000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), -2000, -1000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), -1000, 0)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), 0, 1000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), 1000, 2000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), 2000, 3000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), 3000, 4000)),
                                       float(feature_calculators.range_count(np.asarray(x[1]), 4000, np.inf)),

                                       float(feature_calculators.ratio_value_number_to_time_series_length(x[1])),
                                       float(feature_calculators.first_location_of_minimum(x[1])),
                                       float(feature_calculators.first_location_of_maximum(x[1])),
                                       float(feature_calculators.last_location_of_minimum(x[1])),
                                       float(feature_calculators.last_location_of_maximum(x[1])),

                                       float(feature_calculators.time_reversal_asymmetry_statistic(x[1], 10)),
                                       float(feature_calculators.time_reversal_asymmetry_statistic(x[1], 100)),
                                       float(feature_calculators.time_reversal_asymmetry_statistic(x[1], 1000)),

                                       float(feature_calculators.autocorrelation(x[1], 5)),
                                       float(feature_calculators.autocorrelation(x[1], 10)),
                                       float(feature_calculators.autocorrelation(x[1], 50)),
                                       float(feature_calculators.autocorrelation(x[1], 100)),
                                       float(feature_calculators.autocorrelation(x[1], 1000)),

                                       float(feature_calculators.c3(x[1], 5)),
                                       float(feature_calculators.c3(x[1], 10)),
                                       float(feature_calculators.c3(x[1], 100)),

                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 1, 'attr': 'real'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 1, 'attr': 'imag'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 1, 'attr': 'angle'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 2, 'attr': 'real'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 2, 'attr': 'imag'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 2, 'attr': 'angle'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 3, 'attr': 'real'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 3, 'attr': 'imag'}]))[
                                                 0][1]),
                                       float(list(
                                           feature_calculators.fft_coefficient(x[1], [{'coeff': 3, 'attr': 'angle'}]))[
                                                 0][1]),

                                       float(feature_calculators.longest_strike_above_mean(x[1])),
                                       float(feature_calculators.longest_strike_below_mean(x[1])),
                                       float(feature_calculators.cid_ce(x[1], 0)),
                                       float(feature_calculators.cid_ce(x[1], 1)),

                                       float(feature_calculators.binned_entropy(x[1], 5)),
                                       float(feature_calculators.binned_entropy(x[1], 10)),
                                       float(feature_calculators.binned_entropy(x[1], 20)),
                                       float(feature_calculators.binned_entropy(x[1], 50)),
                                       float(feature_calculators.binned_entropy(x[1], 80)),
                                       float(feature_calculators.binned_entropy(x[1], 100)),

                                       float(feature_calculators.number_crossing_m(x[1], 0)),
                                       float(feature_calculators.number_peaks(x[1], 10)),
                                       float(feature_calculators.number_peaks(x[1], 50)),
                                       float(feature_calculators.number_peaks(x[1], 100)),
                                       float(feature_calculators.number_peaks(x[1], 500)),

                                       float(list(feature_calculators.spkt_welch_density(x[1], [{'coeff': 1}]))[0][1]),
                                       float(list(feature_calculators.spkt_welch_density(x[1], [{'coeff': 10}]))[0][1]),
                                       float(list(feature_calculators.spkt_welch_density(x[1], [{'coeff': 50}]))[0][1]),
                                       float(list(feature_calculators.spkt_welch_density(x[1], [{'coeff': 100}]))[0][1])
                                   ))

df_features = rdd_features.toDF(["idx",
                                 "ttf",
                                 "mean",
                                 "max",
                                 "min",
                                 "std",
                                 "var",
                                 "ptp"
                                 "percentile_10",
                                 "percentile_20",
                                 "percentile_30",
                                 "percentile_40",
                                 "percentile_50",
                                 "percentile_60",
                                 "percentile_70",
                                 "percentile_80",
                                 "percentile_90",

                                 "skew",
                                 "kurtosis",
                                 "kstat_1",
                                 "kstat_2",
                                 "kstat_3",
                                 "kstat_4",
                                 "moment_1",
                                 "moment_2",
                                 "moment_3",
                                 "moment_4",

                                 "abs_energy",
                                 "abs_sum_of_changes",
                                 "count_above_mean",
                                 "count_below_mean",
                                 "mean_abs_change",
                                 "mean_change",
                                 "var_larger_than_std_dev",

                                 "range_minf_m4000",
                                 "range_m4000_m3000",
                                 "range_m3000_m2000",
                                 "range_m2000_m1000",
                                 "range_m1000_0",
                                 "range_0_p1000",
                                 "range_p1000_p2000",
                                 "range_p2000_p3000",
                                 "range_p3000_p4000",
                                 "range_p4000_pinf",

                                 "ratio_unique_values",
                                 "first_loc_min",
                                 "first_loc_max",
                                 "last_loc_min",
                                 "last_loc_max",

                                 "time_rev_asym_stat_10",
                                 "time_rev_asym_stat_100",
                                 "time_rev_asym_stat_1000",

                                 "autocorrelation_5",
                                 "autocorrelation_10",
                                 "autocorrelation_50",
                                 "autocorrelation_100",
                                 "autocorrelation_1000",

                                 "c3_5",
                                 "c3_10",
                                 "c3_100",

                                 "fft_1_real",
                                 "fft_1_imag",
                                 "fft_1_angle",
                                 "fft_2_real",
                                 "fft_2_imag",
                                 "fft_2_angle",
                                 "fft_3_real",
                                 "fft_3_imag",
                                 "fft_3_angle",

                                 "long_strk_above_mean",
                                 "long_strk_below_mean",
                                 "cid_ce_0",
                                 "cid_ce_1",

                                 "binned_entropy_5",
                                 "binned_entropy_10",
                                 "binned_entropy_20",
                                 "binned_entropy_50",
                                 "binned_entropy_80",
                                 "binned_entropy_100",

                                 "num_crossing_0",
                                 "num_peaks_10",
                                 "num_peaks_50",
                                 "num_peaks_100",
                                 "num_peaks_500",

                                 "spkt_welch_density_1",
                                 "spkt_welch_density_10",
                                 "spkt_welch_density_50",
                                 "spkt_welch_density_100"
                                 ])

df_features.show(10)

end = time()
print('Feature generation takes {} seconds'.format(round(end - start, 2)))

# Stop the Spark session
spark.stop()
