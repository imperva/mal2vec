import os
import itertools
import numpy as np
import pandas as pd

def split_sequence_by_time(events_list, timestamps, sequence_timeout):
    # order by timestamp
    timestamps, events_list = zip(*sorted(zip(timestamps, events_list)))

    # get the difference between every 2 events
    time_diff = np.diff(timestamps)
    
    # split if the time difference is larger than the threshold
    where_to_split = np.where(time_diff > 1000 * sequence_timeout)[0]
        
    # for each splitting position:
    # create a sublist and add it to the result
    splitted, prev_pos = [], 0
    for i in where_to_split:
        splitted.append(events_list[prev_pos:i+1])
        prev_pos = i+1
    
    # append all the remaning events
    splitted.append(events_list[prev_pos:])

    return splitted

def create_sentences(df, time_column, event_column, label_column, grouping_columns, timeout=300):
    df[event_column] = df[event_column].astype(str)
    df[time_column] = df[time_column].astype(int)
    grouping_columns.append(label_column)
    
    df.index = df.index.set_names(['mal2vec_index'])
    df = df.reset_index()
    
    print('Step 1/4: Grouping events by {}'.format(grouping_columns))
    # -----------------------------------------------------
    # 1) group events by 'grouping_columns'
    # 2) create 2 lists: event_ids, timestamps
    # 3) split event_ids by time and get the 'true sequeneces'
    agg_df = df.sort_values([time_column], ascending=True) \
        .groupby(grouping_columns) \
        .agg({
            'mal2vec_index': list,
            time_column: list,
        })

    print('Step 2/4: Splitting sequences by timeout={}sec'.format(timeout), flush=True)
    # ----------------------------------------------------------------------
    true_sequences = []
    agg_df.progress_apply(lambda x: split_sequence_by_time(x['mal2vec_index'], x[time_column], timeout), axis=1) \
          .apply(true_sequences.extend)

    print('Step 3/4: Mapping each event to the sequence it belongs to')
    # ------------------------------------------------------
    events, indices = [], []
    for i, values in enumerate(true_sequences):
        events.extend(values)
        indices.extend([i]*len(values))
    mapping_df = pd.DataFrame({'sequence_id': indices, 'mal2vec_index': events})

    ### create aggregation dictionary:
    # for each column in grouping_columns -> select the most common
    most_common_lambda = lambda x: pd.Series.mode(x)[0]
    agg_dict = {k:most_common_lambda for k in grouping_columns}
    # for event_column -> create a list
    agg_dict[event_column] = list
    

    # 1) merge mapping with the original df
    # 2) group by the sequence_id
    sentences_df = mapping_df\
                    .merge(df, on='mal2vec_index', how='inner')\
                    .groupby(['sequence_id']) \
                    .agg(agg_dict)\
                    .rename(columns={event_column: 'sentence', label_column:'label'})\
                    .reset_index()[['sentence', 'label']]

    
    return sentences_df

def dump_sentences(sentences_df, dataset_name):
    print('Step 4/4: Saving sentences to disk')
    # ---------------------------------------
    output_folder = os.path.join(os.getcwd(), 'sentences', dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save global sentecnes file
    sentences_filepath = os.path.join(output_folder, 'sentences.txt')
    
    sentences = sentences_df['sentence'].apply(lambda s: ' '.join([k for k, g in itertools.groupby(s)]))

    with open(sentences_filepath, 'w') as output_file:
        output_file.write('\n'.join(set(sentences)))

    # save dataframe as csv
    sentences_df['sentence'] = sentences_df['sentence'].apply(lambda x: ' '.join(x))
    filepath = os.path.join(output_folder, 'sentences_df.csv')
    sentences_df.to_csv(filepath, index=False)

    print('Done!')
    return sentences_filepath