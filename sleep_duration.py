import mc_package.sleep_periods as sleep_periods
import pandas as pd
MS_PER_DAY = 1000*60*60*24
MS_PER_HOUR = 1000*60*60

def sleep_duration(part_id, start, end):
    sleep_res = sleep_periods.sleep_periods(part_id, start=start, end=end)
    if sleep_res is None:
        return None
    res = sleep_periods.sleep_periods(part_id, start=start, end=end)['values']
    if res is not None:
        on, off, qual, starts = res
        res_df = pd.DataFrame(list(zip(on, off, qual, starts)), columns = ['Onset', 'Offset', 'Quality', 'Hour_Start'])
        res_df['Start'] = res_df['Hour_Start']*MS_PER_HOUR + sleep_periods.time_shift(start, 17)
        res_df['Duration'] = res_df['Offset'] - res_df['Onset']
    else:
        res_df = pd.DataFrame([[None, None, None, None, sleep_periods.time_shift(start, 17)]], columns = ['Onset', 'Offset', 'Quality', 'Duration', 'Start'])
    return res_df[['Onset', 'Offset', 'Quality', 'Duration', 'Start']]
