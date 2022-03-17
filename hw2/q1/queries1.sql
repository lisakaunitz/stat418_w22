select count(*) as "number of movies"
from movies
where number_of_votes > 1000
group by substr(rating, 1, 1); 