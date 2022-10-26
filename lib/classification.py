import numpy as np
from pandas import isnull, read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import lib.config.ds_charts as ds
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, show, subplots, Axes, title
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lib.utils import generate_timestamp, save_image, save_pd_as_csv

def split_train_test_sets(dataset, data, file_tag, target, positive = 1, negative = 0):
    print('[+] Splitting the dataset into training and testing subsets')
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(f'datasets/{file_tag}_train.csv', index=False)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(f'datasets/{file_tag}_test.csv', index=False)
    values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
    values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

    plt.figure(figsize=(12,4))
    ds.multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
    save_image(dataset, 'data_distributions_per_dataset')

def clean_dataset(df):
    assert isinstance(df, DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def perform_naive_bayes_analysis(file_tag, target):
    print('[+] Performing Naive Bayes analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()
    save_pd_as_csv("snp", isnull(train).sum() > 0, "empty_values")

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    clf = BernoulliNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    figure()
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_nb_best_gaussian')
    estimators = {'GaussianNB': GaussianNB(),
 #             'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
#              'CategoricalNB': CategoricalNB
              }

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))

    figure()
    ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    save_image(file_tag, f'{file_tag}_nb_study')


def plot_overfitting_study(dataset, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    ds.multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    save_image(dataset, f'overfitting_{name}')

def perform_knn_analysis(file_tag, target):
    print('[+] Performing KNN analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    eval_metric = accuracy_score
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        y_tst_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
        values[d] = y_tst_values

    figure()
    ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    save_image(file_tag, f'{file_tag}_knn_study')
    print(f'[!] Best results with %d neighbors and %s'%(best[0], best[1]))

    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_knn_best')

    d = 'euclidean'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        prd_trn_Y = knn.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(file_tag, nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))

def perform_decision_trees_analysis(file_tag, target):
    print('[+] Performing decision trees analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_model = None

    fig, axs = subplots(1, 2, squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
        save_image(file_tag, f'{file_tag}_dt_study')

        print('[!] Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    label_values = [str(value) for value in labels]
    #plot_tree(best_model, feature_names=train.columns, class_names=label_values)
    #save_image(file_tag, f'{file_tag}_dt_best_tree')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_dt_best')

    import graphviz
    # DOT data
    dot_data = export_graphviz(best_model, out_file=None, 
                                    feature_names=train.columns,  
                                    class_names=label_values,
                                    filled=True)

    graph = graphviz.Source(dot_data, format="png")
    filename = f'./output/{file_tag}/images/{generate_timestamp()}__best_decision_tree'
    print(f'[+] saving {filename} image as png...') 
    graph.render(filename)

    variables = train.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    ds.horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
    save_image(file_tag, f'{file_tag}_dt_ranking.png')

    imp = 0.0001
    f = 'entropy'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for d in max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(trnX, trnY)
        prdY = tree.predict(tstX)
        prd_tst_Y = tree.predict(tstX)
        prd_trn_Y = tree.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    figure()
    plot_overfitting_study(file_tag, max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))
    save_image(file_tag, f'{file_tag}_dt_ranking')

def perform_random_forest_analysis(file_tag, target):
    print('[+] Performing random forest analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10, 25]
    max_features = [.3, .5, .7, 1]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        
    ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                        xlabel='nr estimators', ylabel='accuracy', percentage=True)
    save_image(file_tag, f'{file_tag}_rf_study')

    print('[!] Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    figure()
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_rf_best')

    variables = train.columns
    importances = best_model.feature_importances_
    stdevs = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    ds.horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
    save_image(file_tag, f'{file_tag}_rf_ranking')

    f = 0.7
    max_depth = 10
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
        rf.fit(trnX, trnY)
        prd_tst_Y = rf.predict(tstX)
        prd_trn_Y = rf.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(file_tag, n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={max_depth}_vars={f}', xlabel='nr_estimators', ylabel=str(eval_metric))
        
def perform_multi_layer_perceptrons(file_tag, target):
    print('[+] Performing multi layer perceptrons analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    lr_type = ['constant', 'invscaling', 'adaptive']
    max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
    learning_rate = [.1, .5, .9]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(lr_type)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    for k in range(len(lr_type)):
        d = lr_type[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in max_iter:
                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)
                mlp.fit(trnX, trnY)
                prdY = mlp.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues
        ds.multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                            xlabel='mx iter', ylabel='accuracy', percentage=True)
    save_image(file_tag, f'{file_tag}_mlp_study')
    print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    figure()
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_mlp_best')

    lr_type = 'adaptive'
    lr = 0.9
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in max_iter:
        mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
        mlp.fit(trnX, trnY)
        prd_tst_Y = mlp.predict(tstX)
        prd_trn_Y = mlp.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    figure()
    plot_overfitting_study(file_tag, max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr episodes', ylabel=str(eval_metric))
    save_image(file_tag, f'{file_tag}_mlp_best')

def perform_gradient_boosting(file_tag, target):
    print('[+] Performing gradient boosting analysis')

    train: DataFrame = read_csv(f'datasets/{file_tag}_train.csv')
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'datasets/{file_tag}_test.csv')
    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10, 25]
    learning_rate = [.1, .5, .9]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in n_estimators:
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = gb
            values[lr] = yvalues
        ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    
    save_image(file_tag, f'{file_tag}_gb_study')
    print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    ds.plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    save_image(file_tag, f'{file_tag}_gb_best')

    variables = train.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    stdevs = np.std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    ds.horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
    save_image(file_tag, f'{file_tag}_gb_ranking.png')

    lr = 0.7
    max_depth = 10
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
        gb.fit(trnX, trnY)
        prd_tst_Y = gb.predict(tstX)
        prd_trn_Y = gb.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(file_tag, n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}', xlabel='nr_estimators', ylabel=str(eval_metric))
