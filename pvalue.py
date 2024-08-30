from scipy import stats

# Provided data
data1 = [162.9837, 186.9597, 238.9167, 163.6407, 181.9682, 154.4328, 164.2836, 202.3357, 139.9143, 177.1744]
data2 = [149.0011,	143.7789,	238.5358,	146.3536,	132.2959,	162.0039,	177.3793,	152.5978,	152.3127,	220.5307]

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(data1, data2, alternative='greater')

print("T-statistic:", t_statistic)
print("P-value:", p_value)