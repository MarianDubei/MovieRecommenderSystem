# MovieRecommenderSystem

The idea behind this project is to implement a system that recommends users new movies they have not watched yet by analyzing their previous interactions.
We chose to use a model-based collaborative approach which involves analyzing a user-movie interaction matrix assuming a latent model. This latent model with factors helps us predict future ratings. We found these latent factors and solved dimensionality issues by matrix factorization. SVD was used as a matrix factorization algorithm. We can get predicted ratings of unwatched films by taking only k most significant latent factors and simply multiply received matrices.
