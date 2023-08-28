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

df_features = df_features.withColumn("mean", F.udf(lambda x: float(np.mean(x)))("ad")) \
.withColumn("max", F.udf(lambda x: float(np.max(x)))("ad")) \
.withColumn("min", F.udf(lambda x: float(np.min(x)))("ad")) \
.withColumn("std", F.udf(lambda x: float(np.std(x)))("ad")) \
.withColumn("var", F.udf(lambda x: float(np.var(x)))("ad")) \
.withColumn("ptp", F.udf(lambda x: float(np.ptp(x)))("ad")) \
.withColumn("percentile_10", F.udf(lambda x: float(np.percentile(x, 10)))("ad")) \
.withColumn("percentile_20", F.udf(lambda x: float(np.percentile(x, 20)))("ad")) \
.withColumn("percentile_30", F.udf(lambda x: float(np.percentile(x, 30)))("ad")) \
.withColumn("percentile_40", F.udf(lambda x: float(np.percentile(x, 40)))("ad")) \
.withColumn("percentile_50", F.udf(lambda x: float(np.percentile(x, 50)))("ad")) \
.withColumn("percentile_60", F.udf(lambda x: float(np.percentile(x, 60)))("ad")) \
.withColumn("percentile_70", F.udf(lambda x: float(np.percentile(x, 70)))("ad")) \
.withColumn("percentile_80", F.udf(lambda x: float(np.percentile(x, 80)))("ad")) \
.withColumn("percentile_90", F.udf(lambda x: float(np.percentile(x, 90)))("ad")) \
\
.withColumn("skew", F.udf(lambda x: float(sp.stats.skew(x)))("ad")) \
.withColumn("kurtosis", F.udf(lambda x: float(sp.stats.kurtosis(x)))("ad")) \
.withColumn("kstat_1", F.udf(lambda x: float(sp.stats.kstat(x, 1)))("ad")) \
.withColumn("kstat_2", F.udf(lambda x: float(sp.stats.kstat(x, 2)))("ad")) \
.withColumn("kstat_3", F.udf(lambda x: float(sp.stats.kstat(x, 3)))("ad")) \
.withColumn("kstat_4", F.udf(lambda x: float(sp.stats.kstat(x, 4)))("ad")) \
.withColumn("moment_1", F.udf(lambda x: float(sp.stats.moment(x, 1)))("ad")) \
.withColumn("moment_2", F.udf(lambda x: float(sp.stats.moment(x, 2)))("ad")) \
.withColumn("moment_3", F.udf(lambda x: float(sp.stats.moment(x, 3)))("ad")) \
.withColumn("moment_4", F.udf(lambda x: float(sp.stats.moment(x, 4)))("ad")) \
\
.withColumn("abs_energy", F.udf(lambda x: float(feature_calculators.abs_energy(x)))("ad")) \
.withColumn("abs_sum_of_changes", F.udf(lambda x: float(feature_calculators.absolute_sum_of_changes(x)))("ad")) \
.withColumn("count_above_mean", F.udf(lambda x: float(feature_calculators.count_above_mean(x)))("ad")) \
.withColumn("count_below_mean", F.udf(lambda x: float(feature_calculators.count_below_mean(x)))("ad")) \
.withColumn("mean_abs_change", F.udf(lambda x: float(feature_calculators.mean_abs_change(x)))("ad")) \
.withColumn("mean_change", F.udf(lambda x: float(feature_calculators.mean_change(x)))("ad")) \
.withColumn("var_larger_than_std_dev", F.udf(lambda x: float(feature_calculators.variance_larger_than_standard_deviation(x)))("ad")) \
\
.withColumn("range_minf_m4000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -np.inf, -4000)))("ad")) \
.withColumn("range_m4000_m3000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -4000, -3000)))("ad")) \
.withColumn("range_m3000_m2000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -3000, -2000)))("ad")) \
.withColumn("range_m2000_m1000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -2000, -1000)))("ad")) \
.withColumn("range_m1000_0", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), -1000, 0)))("ad")) \
.withColumn("range_0_p1000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 0, 1000)))("ad")) \
.withColumn("range_p1000_p2000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 1000, 2000)))("ad")) \
.withColumn("range_p2000_p3000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 2000, 3000)))("ad")) \
.withColumn("range_p3000_p4000", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 3000, 4000)))("ad")) \
.withColumn("range_p4000_pinf", F.udf(lambda x: float(feature_calculators.range_count(np.asarray(x), 4000, np.inf)))("ad")) \
\
.withColumn("ratio_unique_values", F.udf(lambda x: float(feature_calculators.ratio_value_number_to_time_series_length(x)))("ad")) \
.withColumn("first_loc_min", F.udf(lambda x: float(feature_calculators.first_location_of_minimum(x)))("ad")) \
.withColumn("first_loc_max", F.udf(lambda x: float(feature_calculators.first_location_of_maximum(x)))("ad")) \
.withColumn("last_loc_min", F.udf(lambda x: float(feature_calculators.last_location_of_minimum(x)))("ad")) \
.withColumn("last_loc_max", F.udf(lambda x: float(feature_calculators.last_location_of_maximum(x)))("ad")) \
\
.withColumn("time_rev_asym_stat_10", F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 10)))("ad")) \
.withColumn("time_rev_asym_stat_100", F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 100)))("ad")) \
.withColumn("time_rev_asym_stat_1000", F.udf(lambda x: float(feature_calculators.time_reversal_asymmetry_statistic(x, 1000)))("ad")) \
\
.withColumn("autocorrelation_5", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 5)))("ad")) \
.withColumn("autocorrelation_10", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 10)))("ad")) \
.withColumn("autocorrelation_50", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 50)))("ad")) \
.withColumn("autocorrelation_100", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 100)))("ad")) \
.withColumn("autocorrelation_1000", F.udf(lambda x: float(feature_calculators.autocorrelation(x, 1000)))("ad")) \
\
.withColumn("c3_5", F.udf(lambda x: float(feature_calculators.c3(x, 5)))("ad")) \
.withColumn("c3_10", F.udf(lambda x: float(feature_calculators.c3(x, 10)))("ad")) \
.withColumn("c3_100", F.udf(lambda x: float(feature_calculators.c3(x, 100)))("ad")) \
\
.withColumn("fft_1_real", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'real'}]))[0][1]))("ad")) \
.withColumn("fft_1_imag", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'imag'}]))[0][1]))("ad")) \
.withColumn("fft_1_angle", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'angle'}]))[0][1]))("ad")) \
.withColumn("fft_2_real", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'real'}]))[0][1]))("ad")) \
.withColumn("fft_2_imag", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'imag'}]))[0][1]))("ad")) \
.withColumn("fft_2_angle", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'angle'}]))[0][1]))("ad")) \
.withColumn("fft_3_real", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'real'}]))[0][1]))("ad")) \
.withColumn("fft_3_imag", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'imag'}]))[0][1]))("ad")) \
.withColumn("fft_3_angle", F.udf(lambda x: float(list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'angle'}]))[0][1]))("ad")) \
\
.withColumn("long_strk_above_mean", F.udf(lambda x: float(feature_calculators.longest_strike_above_mean(x)))("ad")) \
.withColumn("long_strk_below_mean", F.udf(lambda x: float(feature_calculators.longest_strike_below_mean(x)))("ad")) \
.withColumn("cid_ce_0", F.udf(lambda x: float(feature_calculators.cid_ce(x, 0)))("ad")) \
.withColumn("cid_ce_1", F.udf(lambda x: float(feature_calculators.cid_ce(x, 1)))("ad")) \
\
.withColumn("binned_entropy_5", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 5)))("ad")) \
.withColumn("binned_entropy_10", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 10)))("ad")) \
.withColumn("binned_entropy_20", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 20)))("ad")) \
.withColumn("binned_entropy_50", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 50)))("ad")) \
.withColumn("binned_entropy_80", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 80)))("ad")) \
.withColumn("binned_entropy_100", F.udf(lambda x: float(feature_calculators.binned_entropy(x, 100)))("ad")) \
\
.withColumn("num_crossing_0", F.udf(lambda x: float(feature_calculators.number_crossing_m(x, 0)))("ad")) \
.withColumn("num_peaks_10", F.udf(lambda x: float(feature_calculators.number_peaks(x, 10)))("ad")) \
.withColumn("num_peaks_50", F.udf(lambda x: float(feature_calculators.number_peaks(x, 50)))("ad")) \
.withColumn("num_peaks_100", F.udf(lambda x: float(feature_calculators.number_peaks(x, 100)))("ad")) \
.withColumn("num_peaks_500", F.udf(lambda x: float(feature_calculators.number_peaks(x, 500)))("ad")) \
\
.withColumn("spkt_welch_density_1", F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]))("ad")) \
.withColumn("spkt_welch_density_10", F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]))("ad")) \
.withColumn("spkt_welch_density_50", F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]))("ad")) \
.withColumn("spkt_welch_density_100", F.udf(lambda x: float(list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]))("ad")) \

df_features.show(10)

end = time()
print('Feature generation takes {} seconds'.format(round(end - start, 2)))

# Stop the Spark session
spark.stop()
