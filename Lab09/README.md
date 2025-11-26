# Lab 8

---
b) Efectul lui Y si θ asupra posteriorului pentru n

Când Y creste (de la 0 la 5 sau 10), posteriorul pentru n se muta spre valori
mai mari. Practic, daca vedem mai multi cumparatori, modelul are nevoie de un
numar mai mare de clienti ca sa explice datele, deci creste si n-ul estimat.

Efectul lui θ (probabilitatea de cumparare) este invers:  
– daca θ este mic (0.2), atunci ca sa obtinem acelasi Y avem nevoie de mai
multi clienti, deci posteriorul pentru n se muta spre valori mari.  
– daca θ este mare (0.5), este mai usor sa obtinem Y, deci n poate fi mai mic, astfel posteriorul pentru n scade.

In concluzie:  
- creste Y → creste n posterior  
- creste θ → scade n posterior

---
d) Diferenta dintre predictiva pentru Y* si posteriorul pentru n

Posteriorul lui n ne spune cat de multi clienti credem ca au fost in ziua
observata, tinand cont de prior si de Y. Asta e despre “ziua trecuta”.

Predictiva P(Y*) spune ce ne asteptam sa vedem intr-o zi viitoare, daca
conditia ramane aceeasi. Aici avem doua surse de incertitudine:  
1. n nu il stim sigur (luam probe din posterior),  
2. chiar si daca am sti n, Y* are variatia lui binomiala.

De aceea predictiva este in general mai “lata” si mai raspandita decat
posteriorul lui n. Practic, Y* variaza mai mult decat n, pentru ca e inca un
proces aleator pe langa incertitudinea lui n.
