## Notes for working on Tasks

### Aufgabe 1

##### 1.1 Hyperparameter sweep
ICNN - Input convex neural network
FFNN - Feed forward neural network

Varrieren der Werte: \
> Layers [1, 2, 3] \
> nodes [4, 8, 16]

Ergebnisse sind entsprechend der Layer anzahl und der notes
benannt und in overleaf. 
Was ist sind die besten parameter für 1500 epochen?
- Das beste Ergebnis hat 3 layer und 16 knoten geliefert.
- Die größte abweichen zeigt sich bei verringerung von knoten und layern im
bereich der mittleren nicht-konvexitäten
- reduziert man weiter, weicht auch der rechte bereich ab 0.8 immer weiter ab.

Variation der epochen
- Erhöhung der epochen bei 3L16 bringt kaum eine verbesserung
- Erhöhung bei geringer Zahl an hidden layers bringt keine verbesserung
- Erhöhung der epochen führt augenscheinlich nur bei mehr als einem layer zur
verbesserung.

Ansatzfunktionen:
- relu liefert ergebnisse mit kanten, relativ ungenau
- Ergebnisse von sigmoid unterscheiden sich kaum von softplus
- Welche Aktivierungsfunktionen stecken hinter den namen?

#### 1.2 Input neural networks

Die Badewannenfunktion soll messpunkte eines Physischen 
systems simuliern. Dieses wird als konvex erwartet.
Passe den Code in *models.py* an, so das NN ein 
InputConvexNN wird. 
- Convex activation Function (softplus)
- Convexe und nicht-negative aktivierungsfunktionen in jedem
nicht versteckten layer (Softplus mit nicht-negativen gewichten)
- Convexe und nicht-negativer output layer

Ergebnisse werden mit dem vorherigen Modell in 3 Layer und 16 Knoten
verglichen. Das Ergebnis sieht nicht so gut aus wie die 
Einstellungen aus Aufgabe 1.1. Eine Begründung hierfür fehlt.

### Aufgabe 2
Fragen beantwortet, siehe präsentation

#### 2.1 Non trainable custom layer
Keine ahnung was man hier machen muss
