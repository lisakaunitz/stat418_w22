# Creating Movies table
CREATE TABLE movies (
  movie_title VARCHAR(45) PRIMARY KEY,
  release_year INT,
  genre VARCHAR(45),
  average_rating REAL,
  number_of_votes INT,
  plot_description VARCHAR(255),
  CHECK(
      release_year BETWEEN 1887 AND 2021 AND # adding constraint 1
      average_rating BETWEEN 1 AND 10 # adding constraint 2
    )
  ); 
# Creating Users table
CREATE TABLE Users (
  username VARCHAR(45),
  first_name VARCHAR(45),
  last_name VARCHAR(45),
  PRIMARY KEY (username)
  );
# Creating Movies table
CREATE TABLE Ratings (
  username VARCHAR(45),
  first_name VARCHAR(45),
  movie_title VARCHAR(45),
  rating INT,
  FOREIGN KEY(username) REFERENCES Users(username),
  FOREIGN KEY(movie_title) REFERENCES movies(movie_title),
  CHECK (rating BETWEEN 1 AND 10),
  PRIMARY KEY (username)
  );
