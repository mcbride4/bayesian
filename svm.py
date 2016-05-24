import csv
import math
import numpy as np
from sklearn.svm import SVC
from bayesalgorithm27 import bayes
from bayesian27 import DataHandler


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class ClassifierSVM(object):

    def __init__(self, path):
        self.users = {}
        self.movies = {}
        self.ratings = AutoVivification()
        self.path = path
        self.load_movies()
        self.load_ratings()
        self.load_users()
        self.users_to_calculate = self.users.keys()[0:100]
        self.similarity = AutoVivification()
        self.movies_correlation = AutoVivification()
        self.find_similarities_between_friends()
        self.find_genre_correlation()
        print "after correlation"
        self.x_train_data = AutoVivification()
        self.y_train_data = {}
        self.score = []
        self.prepare_train_data()
        self.avg_score = 0
        print "prepared data"
        self.evaluate_classification()
        self.count_avg_score()

    def load_movies(self):
        with open(self.path + 'movies.csv', "rU") as infile:
            r = csv.reader(infile, delimiter=';')
            for row in r:
                if len(row) > 1:
                    self.movies[row[0]] = row[2:(len(row) - 1)]

    def load_ratings(self):
        with open(self.path + 'ratings.csv', "rU") as infile:
            r = csv.reader(infile, delimiter=';')
            for row in r:
                if len(row) > 1:
                    self.ratings[row[0]][row[1]] = row[2]

    def load_users(self):
        with open(self.path + 'users.csv', "rU") as infile:
            r = csv.reader(infile, delimiter=';')
            for row in r:
                if len(row) > 1:
                    self.users[row[0]] = row[1:(len(row) - 1)]

    @staticmethod
    def sum_differences(friend_ratings, ratings, suma=0):
        for i in range(0, (len(ratings))):
            diff = float(ratings[i]) - float(friend_ratings[i])
            suma = suma + abs(diff)
        return suma

    def find_common_movies(self, friend, ranked_movies):
        ranked_movies_friends = set(self.ratings[friend].keys())
        common_movies = ranked_movies.intersection(ranked_movies_friends)
        return common_movies

    def find_ratings_of_common_movies(self, common_movies, friend, user):
        ratings = []
        friend_ratings = []
        for movie in common_movies:
            ratings.append(self.ratings[user][movie])
            friend_ratings.append(self.ratings[friend][movie])
        return ratings, friend_ratings

    def write_similarity(self, friend, friend_ratings, ratings, user):
        if len(ratings) > 0:
            suma = self.sum_differences(friend_ratings, ratings)
            self.similarity[user][friend] = (1 - suma / len(ratings))
        else:
            self.similarity[user][friend] = 0

    def find_similarities_between_friends(self):
        for user in self.users.keys():
            ranked_movies = set(self.ratings[user].keys())
            for friend in self.users[user][4:(len(self.users[user]) - 1)]:
                common_movies = self.find_common_movies(friend, ranked_movies)
                friend_ratings, ratings = self.find_ratings_of_common_movies(common_movies, friend, user)
                self.write_similarity(friend, friend_ratings, ratings, user)

    def find_genre_correlation(self):
        for movie in self.movies.keys():
            for film in self.movies.keys():
                if film != movie:
                    genres_m = set(self.movies[movie])
                    genres_f = set(self.movies[film])
                    count = len(list(genres_f.intersection(genres_m)))
                    if count > 0:
                        self.movies_correlation[movie][film] = float(count) / len(self.movies[movie])

    def prepare_y_train_data(self, user):
        self.y_train_data[user] = []
        counter = 0
        for f in self.movies.keys():
            while len(self.y_train_data[user]) < int(0.75 * len(self.ratings[user].keys())) and counter < 250:
                counter += 1
                if f not in self.ratings[user].keys():
                    self.y_train_data[user].append(f)
            break
        w = self.ratings[user].keys()[0:int(0.75 * len(self.ratings[user].keys()))]
        seen = set(self.y_train_data[user])
        for value in w:
            if value not in seen:
                self.y_train_data[user].append(value)
                seen.add(value)
        # self.y_train_data.extend(self.ratings[user].keys()[0:int(0.75 * len(self.ratings[user].keys()))])
        # print self.y_train_data

    def append_info_about_films_to_x(self, film, user):
        for movie in self.ratings[user].keys():
            mark = float(self.ratings[user][movie])
            correlation = self.movies_correlation[movie][film]
            if isinstance(correlation, float):
                if isinstance(mark, float):
                    self.x_train_data[user][film].append(mark * correlation)
                else:
                    self.x_train_data[user][film].append(correlation)
            else:
                if isinstance(mark, float):
                    self.x_train_data[user][film].append(mark)
                else:
                    self.x_train_data[user][film].append(0)

    def append_similarities_to_x_train_data(self, film, user):
        for friend in self.similarity[user].keys():
                self.x_train_data[user][film].append(self.similarity[user][friend])

    def prepare_x_train_data(self, user):
        for film in self.y_train_data[user]:
            self.x_train_data[user][film] = self.users[user][0:3]
            self.append_info_about_films_to_x(film, user)
            self.append_similarities_to_x_train_data(film, user)

    def prepare_train_data(self):
        for user in self.users_to_calculate:
            self.prepare_y_train_data(user)
            self.prepare_x_train_data(user)

    def evaluate_classification(self):
        for user in self.users_to_calculate:
            y_data = self.y_train_data[user]
            y_train = []
            X = []
            for movie in y_data:
                if isinstance(self.ratings[user][movie], str):
                    y_train.append(1)
                else:
                    y_train.append(0)
                X.append(map(float, self.x_train_data[user][movie]))
            x_train = [x for x in X if x]
            # print x_train
            # print y_train
            print "Number of watched films: ", sum(y_train)
            bayesian = bayes(x_train, y_train)
            bayesian.dh.splitData(0.33)		
            print('Split {0} rows into train={1} and test={2} rows'.format(len(bayesian.dh.dataset), len(bayesian.dh.trainData), len(bayesian.dh.testData)))
            bayesian.dh.separateByClass(bayesian.dh.dataset, bayesian.dh.predictions)
            bayesian.dh.statsByClass()
            bayesian.predictions = bayesian.getPredictions(bayesian.dh.stats, bayesian.dh.testData)
            print('predictions: '.format(bayesian.predictions))
            score = bayesian.getAccuracy(bayesian.dh.testDataPredictions, bayesian.predictions)
            self.score.append(score)
            print "score: ", score

    def count_avg_score(self):
        self.avg_score = sum(self.score)/len(self.score)


if __name__ == '__main__':
    c = ClassifierSVM('/home/mcbride/git/SocialCommunicationNetworks2/ml-1m/output/')
    print c.avg_score
