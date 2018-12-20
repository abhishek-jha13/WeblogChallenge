from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
#############################################################################
# Referred follwing websites for derving results 							#
#	http://www.janvsmachine.net/2017/09/sessionization-with-spark.html      #
#	https://blog.modeanalytics.com/finding-user-sessions-sql/               #
#############################################################################

# session window time in seconds (10 mins)
maxSessionDuration = 600 

#schema of the input file
schema = StructType([StructField("timestamp", StringType(), True) ,StructField("elb", StringType(), True) ,StructField("client_ip_port", StringType(), True) ,StructField("backend_ip_port", StringType(), True) ,StructField("request_processing_time", StringType(), True) ,StructField("backend_processing_time", FloatType(), True) ,StructField("response_processing_time", FloatType(), True) ,StructField("elb_status_code", IntegerType(), True) ,StructField("backend_status_code", IntegerType(), True) ,StructField("received_bytes", IntegerType(), True) ,StructField("sent_bytes", IntegerType(), True) ,StructField("request", StringType(), True) ,StructField("user_agent", StringType(), True) ,StructField("ssl_cipher", StringType(), True) ,StructField("ssl_protocol", StringType(), True)])

# Read input file stored in default location in hdfs into a spark dataframe
weblog_df = spark.read.schema(schema).option("delimiter", ' ').csv('2015_07_22_mktplace_shop_web_log_sample.log') 

# Add column with previous activity timestamp
weblog_df_with_prev_ts_df = weblog_df.select('client_ip_port', 'request', col('timestamp').cast('timestamp').alias('current_timestamp'),lag(	col('timestamp').cast('timestamp'), 1).over(Window.partitionBy('client_ip_port').orderBy(col('timestamp').cast('timestamp'))).alias('prev_timestamp')

# Adds a column specifying whether the current_timestamp is a new session which happens when current_timestamp - prev_timestamp > maxSessionDuration
weblog_df_with_prev_ts_with_new_sess_flag_df = weblog_df_with_prev_ts_df.select('client_ip_port', 'request', 'current_timestamp', 'prev_timestamp',when(unix_timestamp(col('current_timestamp')) - unix_timestamp(col('prev_timestamp')) < lit(maxSessionDuration), lit(0)).otherwise(lit(1)).alias('is_new_session'))

# Giving a session_id to al the requests
weblog_df_sessionized = weblog_df_with_prev_ts_with_new_sess_flag_df.select('client_ip_port', 'current_timestamp', 'request',sum('is_new_session').over(Window.partitionBy('client_ip_port').orderBy('current_timestamp')).alias('session_id'))


#weblog_df_sessionized.show(100, truncate=False)
#weblog_df_sessionized.rdd.count()


############################           Solutions           #####################################################



########################################################Q1: Sessionize the web log by IP ##################################################################
                                                                                                                                                          
weblog_df_session_start_end_times = weblog_df_sessionized.groupBy("client_ip_port", "session_id").agg(min("current_timestamp")                            
                                       .alias("start_time"),max("current_timestamp").alias("end_time"),count("*").alias("no_url_hits"))  
# Write the results in Solutions1 directory in the default hdfs location                                                                                  
weblog_df_session_start_end_times.write.csv('Solution1')
###########################################################################################################################################################


########################################################Q2: Average Session Time   #######################################################################

weblog_df_session_start_end_times.createOrReplaceTempView("weblog_df_session_start_end_times")

weblog_df_avg_session_time = spark.sql('select client_ip_port, avg(unix_timestamp(end_time) - unix_timestamp(start_time)) \
                                        as avg_session_time from weblog_df_session_start_end_times group by 1')
weblog_df_session_start_end_times.write.csv('Solution2')
#weblog_df_avg_session_time.show(100, truncate=False)
###########################################################################################################################################################

########################################################Q3: Unique URL visits per session##################################################################
weblog_df_sessionized.createOrReplaceTempView("weblog_df_sessionized")

client_ip_port_unique_url_visted_df = spark.sql('select client_ip_port, session_id, count(distinct request) as count_unique_url_visited \
													from weblog_df_sessionized group by 1, 2 order by 3 desc')
#client_ip_port_unique_url_visted_df.show(100, truncate=False)
client_ip_port_unique_url_visted_df.write.csv('Solution3')
###########################################################################################################################################################

########################################################Q4: Most engaged user i.e. IPs with maximum session times##########################################

weblog_df_session_start_end_times.createOrReplaceTempView("weblog_df_session_start_end_times")
client_ip_port_df_max_session_duration = spark.sql('select client_ip_port, max(unix_timestamp(end_time) - unix_timestamp(start_time))   \
														as maximum_session_duration  from weblog_df_session_start_end_times group by 1 order by 2 desc')
client_ip_port_df_max_session_duration.show(100, truncate=False)
client_ip_port_df_max_session_duration.write.csv('Solution4')
###########################################################################################################################################################

#########################################################ML Related Questions############################################################################## 
##################### Predict the expected load (requests/second) in the next minute using linear regression of last minute requests/second################
#I tried using time series approach but I did not find any pre-installed library in pyspark. So I used Simple linear regression approach 
weblog_df_with_prev_ts_with_new_sess_flag_df.createOrReplaceTempView("weblog_df_with_prev_ts_with_new_sess_flag_df")
requests_per_minute_df = spark.sql('select date_hour_minute as current_date_hour_minute, lag(date_hour_minute, 1) \
										over(order by date_hour_minute) as prev_date_hour_minute, lag(requests_count, 1) \
										over(order by date_hour_minute) as prev_requests_count, requests_count as current_requests_count \
										from (select from_unixtime(unix_timestamp(current_timestamp), "y-MM-dd hh:mm") as date_hour_minute  \
										, count(request) as requests_count from weblog_df_with_prev_ts_with_new_sess_flag_df group by 1 order by 1)')

#requests_per_minute_df.show(500, truncate=False)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
vectorAssembler = VectorAssembler(inputCols = ['prev_requests_count'], outputCol = 'features')
vrequests_per_minute_df = vectorAssembler.transform(requests_per_minute_df.na.drop())
vrequests_per_minute_df = vrequests_per_minute_df.select(['features', 'current_requests_count'])
vrequests_per_minute_df.show()
splits = vrequests_per_minute_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
lr = LinearRegression(featuresCol = 'features', labelCol='current_requests_count', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
requests_per_minute_df.stat.corr('current_requests_count', 'prev_requests_count')


lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","current_requests_count","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="current_requests_count",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
##########################################################################################################################################################

###########################################################Predict the session length for a given IP######################################################
#Average session length of a given IP
weblog_df_session_start_end_times.createOrReplaceTempView("weblog_df_session_start_end_times")
expected_session_length_client_ip_port_df = spark.sql('select client_ip_port, avg(unix_timestamp(end_time) - unix_timestamp(start_time)) as expected_session_length from weblog_df_session_start_end_times group by 1')


##########################################################  Predict the number of unique URL visits by a given IP ########################################
# The number of unique urls the IP has visited till now.
weblog_df.createOrReplaceTempView("weblog_df")
client_ip_port_unique_urls_visited_df = spark.sql('select client_ip_port, count(distinct request) as unique_urls_visited from weblog_df group by 1 order by 2 desc').show(truncate=False)

##########################################################################################################################################################









