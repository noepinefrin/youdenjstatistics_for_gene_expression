import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import confusion_matrix

class YoudensJStatistics:

    """
    Usage Of Youden's J Statistics Class
    ------------------------------------

    To use this class effectively, you can follow these steps;

    Firstly, parse your data as whole data is represents data, first class of your data (also known as data_healthy), and second class of your data (a.k.a data_cancer) and true labels (data type must be list, ex: [0, 1, 1, 0...])

    Then, initialize your class;

    >>> youden = YoudensJStatistics(data = luad_data, data_healthy = luad_data_healthy, data_cancer = luad_data_cancer, true_labels = true_labels)

    After initializing, you can derive youden values based on the given number of cutoffs according to feature or gene index, respectively. (If you want, you can save as csv that DataFrame.)

    >>> youden_prediction = youden.predict(gene = 0, number_of_cutoff = 100, save_csv = True) # Returns -> Gene Metrics DataFrame, DataFrame Structure example is given below.

    >>> youden_prediction
        CutOff  TN  FP  FN  TP  YJS_Score
    0   11.9    0   28  31  533 0.525424
    1   16.7    214 0   59  319 0.598499
    .
    .
    .

    Also you can get best cutoff interval. This function prints best interval, also return the cutoff values and their Youden's J Statistics scores.

    >>> best_interval = youden.best_interval(gene = 0, youden_prediction) # Returns -> tuple, first index is Youden's J Statistics Score, second index is beginning of the interval, third index is last element of the interval.
    "Best Youden's J Statistics Scores for this AGER is 0.97, and their cutoff interval is 227.39 to 239.11."
    (0.97, 227.39, 239.11)

    You can create a graph using your predicted values.
    In this graph, the X axis will represent the cutoff values, while the Y axis will represent the Youden's J Statistics scores.
    In the graph, the maximum value will be annotate to an appropriate part of the graph on a tile. If you want, you can also set a threshold value in the graph.

    >>> youden.draw(gene = 0, youden_prediction, threshold = 0.9, save_fig = True)
    .../noepinefrin/AGER_youdensjstatistics_score.png
    """

    def __init__(self, data_path: str, data_healthy_path: str, data_cancer_path: str, true_labels_path: str) -> None:
        self.data = pd.read_csv(data_path, dtype=np.float16)
        self.healthy = pd.read_csv(data_healthy_path, dtype=np.float16)
        self.cancer = pd.read_csv(data_cancer_path, dtype=np.float16)
        self.true_labels = pd.read_csv(true_labels_path, dtype=np.int8).values.ravel()
        self.label_order = np.reshape([[[1, 0] if self.healthy.describe().iloc[:, i][5] > self.cancer.describe().iloc[:, i][5] else [0, 1] for _ in range(1)]
                             for i in range(len(self.healthy.columns))], (len(self.data.columns), 2))

    def __calculate_youden_j_statistics(self, predicted_labels: list) -> tuple:
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(self.true_labels, predicted_labels).ravel()

        youden_j_statistics_value = (true_positive / (true_positive + false_negative)) + (true_negative / (true_negative + false_positive)) - 1

        return true_negative, false_positive, false_negative, true_positive, youden_j_statistics_value

    def predict(self, gene: int, number_of_cutoff: int, save_csv_path: str = None) -> pd.DataFrame:

        gene_metrics = pd.DataFrame()

        described_data = self.data.describe()

        third_quartile = described_data.iloc[6, :][gene]
        minimum = described_data.iloc[3, :][gene]
        median = described_data.iloc[5, :][gene]

        accrual = (third_quartile - minimum) / median
        cutoff = described_data.iloc[3, :][gene]

        print(f'This results belong to {self.data.columns[gene]}.')

        for _ in range(number_of_cutoff):

            prediction_list = []

            for index in range(self.data.shape[0]):

                if cutoff > self.data.iloc[:, gene][index]:
                    prediction_list.append(self.label_order[gene][0])

                else:
                    prediction_list.append(self.label_order[gene][1])

            metrics = self.__calculate_youden_j_statistics(prediction_list)

            gene_metrics = gene_metrics.append(

                {
                    'Cutoff': cutoff,
                    'True_Negative': metrics[0],
                    'False_Positive': metrics[1],
                    'False_Negative': metrics[2],
                    'True_Positive': metrics[3],
                    'YJS_Score': metrics[4]
                },

                ignore_index = True

            )

            cutoff += accrual

        if isinstance(save_csv_path, str):
            gene_metrics.to_csv(f'{save_csv_path}/{self.data.columns[gene]}_gene_metrics.csv', index = False)

        return gene_metrics

    def _best_interval(self, gene: int, prediction_df: pd.DataFrame) -> tuple:
        cutoff = prediction_df.iloc[:, 0].values
        youdens_score = prediction_df.iloc[:, 5].values

        unsorted_md_array = np.column_stack((cutoff, youdens_score))
        sorted_md_array = sorted(unsorted_md_array, key=lambda item: item[1])

        interval_where = np.argwhere(sorted_md_array == max(sorted_md_array, key=lambda item: item[1]))

        best_youdens_score = sorted_md_array[interval_where[0][0]][1]
        interval_begin = sorted_md_array[interval_where[0][0]][0]
        interval_last = sorted_md_array[interval_where[-1][0]][0]

        print(f"Your best Youden's J Statistics Score for {self.data.columns[gene]} is {round(best_youdens_score, 3)} and their cutoff interval is {round(interval_begin, 3)} to {round(interval_last, 3)}.")

        return best_youdens_score, interval_begin, interval_last

    def mannwhitneyu(self, save_csv_path: str = None) -> pd.DataFrame:

        MannWhitneyU_pd = pd.DataFrame()

        for gene_index in range(self.data.shape[1]):

            score, p_value = stats.mannwhitneyu(x = self.healthy.iloc[:, gene_index], y = self.cancer.iloc[:, gene_index])

            MannWhitneyU_pd = MannWhitneyU_pd.append(

                {
                    'GENEs': self.data.columns[gene_index],
                    'Mann-Whitney U Scores': score,
                    'P-Values': p_value
                },

                ignore_index = True

            )

        if save_csv_path:
            MannWhitneyU_pd.to_csv(f'{save_csv_path}/MannWhitneyU.csv', index = False)

        return MannWhitneyU_pd


    def __annot_max(self, X, Y, gene: int, prediction_df: pd.DataFrame, ax = None, box_location: tuple = (0.65, 0.975)):
        xmax = X[np.argmax(Y)]
        best_interval = self._best_interval(gene = gene, prediction_df= prediction_df)
        ymax = max(Y)
        text= "PEAK POINT\nCutoff (as a interval): {:.3f} to {:.3f}\nYouden's J Statistics Score: {:.3f}".format(xmax, best_interval[2], ymax)
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle = "square,pad=0.75", fc = "#FCFFE7", ec = "#272932", lw = 1, color = "#272932")
        arrowprops = dict(arrowstyle = "-|>", connectionstyle = "angle, angleA = 0, angleB = 60", linewidth = 2, color = '#272932')
        kw = dict(xycoords = 'data', textcoords = "axes fraction",
                arrowprops = arrowprops, bbox = bbox_props, ha = "left", va = "top", size = 25, color = '#272932', linespacing=2.5)

        ax.annotate(text, xy = (xmax, ymax), xytext = (box_location[0], box_location[1]), **kw)

    def draw(self, gene: int, prediction_df: pd.DataFrame, save_fig_path: str = None):

        plt.style.use('seaborn-deep')

        cutoff = prediction_df.iloc[:, 0].values
        youdens_score = prediction_df.iloc[:, 5].values

        self.__annot_max(X = cutoff, Y = youdens_score, gene = gene, prediction_df=prediction_df)

        plt.rcParams["figure.figsize"] = (35, 35)

        plt.title(f"$\it[{self.data.columns[gene]}]$\nYouden's J Statistics Scores | Cutoff Values", fontsize = 42, pad = 35)
        plt.xlabel("Cutoff Values", size = 40, labelpad = 30)
        plt.ylabel("Youden's J Statistics Score", size = 40, labelpad = 30)


        plt.xticks(fontsize = 35)
        plt.yticks(fontsize = 35)

        plt.scatter(cutoff, youdens_score, marker = 'D', s = 200, alpha = 0.3, color = '#3A9188')

        if save_fig_path:
            plt.savefig(f'{save_fig_path}/{self.data.columns[gene]}_cutoff_youdens.png')

        plt.show()