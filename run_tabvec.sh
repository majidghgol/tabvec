rm -rf ~/Desktop/elicit_data_tabvecs/
cd embedding/
python TableEmbedding.py ~/Desktop/elicit_data/ ~/Desktop/elicit_data_wordembeddings.txt ~/Desktop/elicit_data_tabvecs/
cd clustering/
python TableClustering.py ~/Desktop/elicit_data_tabvecs/ ~/Desktop/elicit_data_clusters.jl kmeans 2
cd ../../toolkit/
python tablejl_to_html.py ~/Desktop/elicit_data_clusters.jl ~/Desktop/elicit_data_clusters.htm