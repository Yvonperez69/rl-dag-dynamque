# Reinforcement learning for automated code generation

L'ambition et la motivation de ce repo est purement pédagogique.


Agent Brook V0 :

Reward uniquement basé sur un llm juge 

Ambition de Brook V1 :

Reward combinaison de :
- LLM juge 
- R_test : reward binaire en fonction de la réussite ou non du test
- R_execution : nb_test, temps d'éxécution, conso mémoire
- R_structure : compléxité, longueur du code

Reward composé avec pondération dynamique , ie apprendre à l'agent à trouver la meilleur pondération ?

