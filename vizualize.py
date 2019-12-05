import os
import mpld3
import gensim
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

def plot_model_results(history, report, df_cm, classes):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    ax[0].plot(history.history['loss'], '--b', color='red', label='Training')
    ax[0].plot(history.history['val_loss'], '--b', color='blue', label='Validaion')
    ax[0].legend(loc='upper right')
    ax[0].title.set_text('Loss (Lower is better)')
    ax[0].set_xlabel('Epoch')

    acc = history.history['acc'] if 'acc' in history.history else history.history['accuracy']
    ax[1].plot(acc, '--b', color='red', label='Training')
    val_acc = history.history['val_acc'] if 'val_acc' in history.history else history.history['val_accuracy']
    ax[1].plot(val_acc, '--b', color='blue', label='Validaion')
    ax[1].legend(loc='lower right')
    ax[1].title.set_text('Accuracy (Higher is better)')
    ax[1].set_xlabel('Epoch')
    fig.tight_layout()

    df_cm = (df_cm / df_cm.astype(np.float).sum(axis=1, keepdims=True)).round(2)
    fig, ax = plt.subplots(figsize=(15,15))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, xticklabels=classes, yticklabels=classes,
               annot=True, fmt='g', annot_kws={"size": 16}, linewidths=.5, ax=ax)
    plt.show()

    print(report)
    

def draw_model(df, sentences_filepath, event_col, cat_col, dataset_name, vector_size=30, epochs=50):
    print('1/3: Extracting embedding Doc2Vec (VECTOR_SIZE: {}; EPOCHS: {})'.format(vector_size, epochs))
    sentences = list(gensim.models.word2vec.LineSentence(sentences_filepath))

    word2vec_model = gensim.models.Word2Vec(sentences,
                                            iter=50,
                                            size=30,
                                            window=5,
                                            min_count=1,
                                            workers=10)


    output_folder = os.path.join(os.getcwd(), 'models', dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model_filepath = os.path.join(output_folder, 'word2vec_model.h5')
    word2vec_model.save(model_filepath)

    events_df = df.groupby([event_col, cat_col]).size().reset_index(name='count').rename(columns={event_col: 'event', cat_col: 'category'})

    color_pallet = [v for k,v in mcolors.CSS4_COLORS.items() if 'light' in k]
    cat2col = {cat: color_pallet[cat_idx] for cat_idx, cat in enumerate(sorted(events_df['category'].unique()))}

    # normalize size
    MIN_MARKER_SIZE, MAX_MARKER_SIZE = 1, 20
    BASE = np.power(max(events_df['count']), 1.0 / (MAX_MARKER_SIZE-MIN_MARKER_SIZE))
    events_df['size'] = events_df['count'].apply(lambda x: MIN_MARKER_SIZE + int(np.log(x) / np.log(BASE)))

    # run t-SNE and extract x,y
    words = list(word2vec_model.wv.vocab.keys())
    vecs = [word2vec_model.wv[word] for word in words]
    events_df['vec_index'] = events_df['event'].apply(word2vec_model.wv.index2word.index)

    print("2/3: Running TSNE...", end='')
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=300, random_state=23)
    new_vecs_2d = tsne_model.fit_transform(vecs)
    print(' - mapped {} vectors'.format(new_vecs_2d.shape[0]))

    X, Y = zip(*new_vecs_2d)
    events_df['x'] = events_df['vec_index'].apply(lambda i: X[i])
    events_df['y'] = events_df['vec_index'].apply(lambda i: Y[i])
    events_df['coordinates'] = events_df['x'].round(2).astype(str) + ' / ' + events_df['y'].round(2).astype(str)

    # create labels
    labels_df = events_df[['event', 'category', 'count', 'size', 'x', 'y', 'coordinates']].copy()

    CSS = """
    table{border-collapse: collapse;}
    th{color: #ffffff;background-color: #000000;}
    td{background-color: #cccccc;}
    table, th, td{
      font-family:Arial, Helvetica, sans-serif;
      font-size: 15;
      border: 1px solid black;
      text-align: right;
    }"""

    print("3/3: Drawing graph")
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.grid(True, alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for i, row in labels_df.iterrows():
        word = row['event']
        neigh_pairs = word2vec_model.wv.most_similar(word, topn=5)
        neigh_words, neigh_sim = zip(*neigh_pairs)
        tooltip_df = labels_df.merge(
            pd.DataFrame({'event': neigh_words, 'similarity': neigh_sim}),
            on='event', 
            how='inner'
        ).sort_values('similarity', ascending=False).drop(columns=['x', 'y'])

        tooltip_df.index = ['THIS'] + ['N{}'.format(i) for i in range(1, len(tooltip_df))]
        tooltip = str(tooltip_df.T.to_html())

        color = cat2col[row['category']]
        cnt_points = ax.plot(row['x'], row['y'], 'o', color=color, mec='k', mew=1, alpha=.95, markersize=row['size'])

        plt_tooltip = mpld3.plugins.PointHTMLTooltip(cnt_points[0], [tooltip], voffset=10, hoffset=10, css=CSS)
        mpld3.plugins.connect(fig, plt_tooltip)

    title_str = 'Embedding of Events'
    ax.set_title(title_str, size=50)
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])

    output_folder = os.path.join(os.getcwd(), 'htmls', dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.join(output_folder, 't-SNE.html')
    mpld3.save_html(fig, open(filename, 'w'))
    plt.close('all')
    
    retlative_path = filename.replace(os.getcwd()+'/', '')
    return retlative_path