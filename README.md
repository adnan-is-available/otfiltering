# <u><b>Collaborative Filtering with Optimal Transport</b></u>

#### Réalisé par Romain Laurent, Félix Breton et Adnan Ben Mansour

Pour plus de détails, cf. le rapport [ici](./notes.pdf). 

Pour faire fonctionner le code, les modules python ``Numpy``, ``Scipy``, ``Sklearn`` et ``Pandas`` sont nécessaires.  
De plus il faut récupérer les datasets ml-1m et ml-25m disponibles à [cette adresse](https://grouplens.org/datasets/movielens/) et les extraire à la racine du projet.

#### Les grandes lignes
Dans ce projet on cherche à estimer si des films ont été vus ou non par des utilisateurs en s'appuyant sur une base de données : MovieLens, et des méthodes issus du Transport Optimal. 

On adapte ces mêmes méthodes pour estimer les notes que les utilisateurs attribuent à ces films. 