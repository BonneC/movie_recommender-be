import pandas as pd
import services.recommender_functions as rec


# calculates recommendation for given ids
def calculate(ids, input_movies):
    # get the titles for the current chunk of ids
    titles = rec.get_titles_from_ids(ids)
    # get 100 movies based on keyword similarity
    keyword_movies = rec.keyword_recommender(titles)
    # narrow down the 100 movies to 20 based on genres
    genre_recs = rec.genre_recommender(input_movies, keyword_movies)
    # set up a popularity rating column
    weighted_movies = rec.set_weighted_rating(genre_recs)
    return weighted_movies


# get recommendations for the given movies
def results(movies):
    # create empty dataframe to which we will append every recommendations per 10 movies
    final = pd.DataFrame()
    # the magic
    correlated = rec.group_movies(movies['user_titles'])
    while True:
        # if there's no more movies to go through, end the loop
        if len(correlated.index) == 0:
            break
        # calculate recommendations for the last chunk of the movies
        if len(correlated.index) <= 10:
            gr_movies = movies[movies['id'].isin(correlated.index)]
            recommendation = calculate(correlated.index, gr_movies)
            # add recommendations to the dataframe
            final = final.append(recommendation)
            break
        #
        grouped = correlated.sort_values(by=[correlated.index[0]], ascending=False)
        # take 10 user movies
        grouped = grouped[:10]
        # calculate recommendations using the indexes for the movies
        gr_ids = grouped.index
        gr_movies = movies[movies['id'].isin(gr_ids)]
        recommendation = calculate(gr_ids, gr_movies)
        # add recommendations to the dataframe
        final = final.append(recommendation)
        # drop the user movies that we already used
        for i in gr_ids:
            correlated.drop(i, axis=0, inplace=True)
            correlated.drop(i, axis=1, inplace=True)
    return final
