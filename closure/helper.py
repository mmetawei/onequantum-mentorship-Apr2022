import numpy as np
from lambeq import BobcatParser, Rewriter, spiders_reader
from lambeq import remove_cups
from lambeq import AtomicType, IQPAnsatz
from lambeq import NumpyModel, Model
from collections.abc import Mapping
from lambeq import CircuitAnsatz
from typing import Any, Callable, Optional

from discopy.quantum.circuit import (Circuit, Discard, Functor, Id,
                                     IQPansatz as IQP, qubit)
from discopy.quantum.gates import Bra, Ket, Rx, Rz
from discopy.rigid import Box, Diagram, Ty
from discopy import Word

from lambeq.ansatz import Symbol, BaseAnsatz
from lambeq.ansatz.circuit import _ArMapT
from discopy.rigid import Diagram, Functor
from lambeq import TketModel

class CircuitAnsatz(BaseAnsatz):
    """Base class for circuit ansatz."""
    def __init__(self, ob_map: Mapping[Ty, int], **kwargs: Any) -> None:
        """Instantiate a circuit ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`discopy.rigid.Ty` to the number of
            qubits it uses in a circuit.
        **kwargs : dict
            Extra parameters for ansatz configuration.

        """
        self.ob_map = ob_map
        self.functor = Functor({}, {})

    def __call__(self, diagram: Diagram) -> Circuit:
        """Convert a DisCoPy diagram into a DisCoPy circuit."""
        return self.functor(diagram)


    def _ob(self, pg_type: Ty) -> int:
        """Calculate the number of qubits used for a given type."""
        return sum(self.ob_map[Ty(factor.name)] for factor in pg_type)

    def _special_cases(self, ar_map: _ArMapT) -> _ArMapT:
        """Convert a DisCoPy box into a tket Circuit element"""
        return ar_map
class TrainAnsatz(CircuitAnsatz):
    # A new Ansatz Design to run the model training
    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False,
                 special_cases: Optional[Callable[[_ArMapT], _ArMapT]] = None):
        super().__init__(ob_map=ob_map, n_layers=n_layers,
                         n_single_qubit_params=n_single_qubit_params)

        if special_cases is None:
            special_cases = self._special_cases

        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.discard = discard
        self.functor = Functor(ob=self.ob_map,
                               ar=special_cases(self._ar))
        
    def _ar(self, box: Box) -> Circuit:
        label = self._summarise_box(box)
        dom, cod = self._ob(box.dom), self._ob(box.cod)

        n_qubits = max(dom, cod)
        n_layers = self.n_layers
        n_1qubit_params = self.n_single_qubit_params

        if n_qubits == 0:
            circuit = Id()
        elif n_qubits == 1:
            syms = [Symbol(f'{label}_{i}') for i in range(n_1qubit_params)]
            rots = [Rx, Rz]
            circuit = Id(qubit)
            for i, sym in enumerate(syms):
                circuit >>= rots[i % 2](sym)
        else:
            n_params = n_layers * (n_qubits-1)
            syms = [Symbol(f'{label}_{i}') for i in range(n_params)]
            params: np.ndarray[Any, np.dtype[Any]] = np.array(syms).reshape(
                    (n_layers, n_qubits-1))
            circuit = IQP(n_qubits, params)

        if cod > dom:
            circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
        elif self.discard:
            circuit >>= Id(cod) @ Discard(dom - cod)
        else:
            circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
        return circuit

def read_data(filename: str):
    labels, sentences1, sentences2 = [], [], []
    with open(filename) as f:
        for line in f:
            split_list = line.split(",")
            t = int(split_list[2])
            labels.append(t)
            sentences1.append(split_list[0])
            sentences2.append(split_list[1])
    return labels, sentences1, sentences2


def label_data(train_data, food_keywords, code_keywords, positive_keywords):
    tag = []
    a = 0
    b = 0
    for i in range(np.size(train_data)):
        words = [word for word in train_data[i].split()]
        for l in words:
            for j in food_keywords:
                if l == j:
                    a = 0
                    b = 1

            for k in code_keywords:
                if l == k:
                    a = 1
                    b = 0


        tag = [a, b]

    return tag


def parse_sentences(train_data, test_data):
    parser = BobcatParser(verbose='text', root_cats=['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]'])

    raw_train_diagrams = parser.sentences2diagrams(train_data)
    raw_test_diagrams = parser.sentences2diagrams(test_data)
    return raw_train_diagrams, raw_test_diagrams


def optimize_diagrams(train_diagrams, test_diagrams):
    train_cup_diagrams = [remove_cups(diagram) for diagram in train_diagrams]
    test_cup_diagrams = [remove_cups(diagram) for diagram in test_diagrams]
    return train_cup_diagrams, test_cup_diagrams


def generate_circuits(train_diagrams, test_diagrams):
    ansatz = TrainAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                       n_layers=1, n_single_qubit_params=3)

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]

    test_circuits = [ansatz(diagram) for diagram in test_diagrams]

    return train_circuits, test_circuits


def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.dot(norm_x, norm_y.T)


class ClassificationModel(NumpyModel):
    right_prediction = 0
    wrong_prediction = 0
    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
        """Return the exact prediction for each diagram.
        Parameters
        ----------
        diagrams : list of :py:class:`~discopy.tensor.Diagram`
            The :py:class:`Circuits <discopy.quantum.circuit.Circuit>`
            to be evaluated.
        Raises
        ------
        ValueError
            If `model.weights` or `model.symbols` are not initialised.
        Returns
        -------
        np.ndarray
            Resulting array.
        """
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`NumpyModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        if self.use_jit:
            lambdified_diagrams = [self._get_lambda(d) for d in diagrams]
            if len(lambdified_diagrams) == 0:
                raise Exception("Lambified list empty !")
            #print(self.weights)
            return np.array([diag_f(*self.weights)
                                for diag_f in lambdified_diagrams])

        diagrams = self._fast_subs(diagrams, self.weights)
        with Tensor.backend('numpy'):
            results = []
            for d in diagrams:
                result = tn.contractors.auto(*d.to_tn()).tensor
                # square amplitudes to get probabilties for pure circuits
                if not d.is_mixed:
                    result = np.abs(result) ** 2
                results.append(self._normalise_vector(result))
            return np.array(results)
    
    def forward(self, x: list[[Diagram, Diagram]]) -> np.ndarray:
        s1_diagrams = []
        s2_diagrams = []
        n_rows = len(x)
        for s1d, s2d in x:
            s1_diagrams.append(s1d)
            s2_diagrams.append(s2d)
        if not s1_diagrams:
            raise Exception("Empty input senetence1  diagrams list!")
        if not s2_diagrams:
            raise Exception("Empty input senetence2  diagrams list!")
        s1_output = self.get_diagram_output(s1_diagrams)
        s2_output = self.get_diagram_output(s2_diagrams)
        #print(s1_output)
        s1_output = s1_output.reshape((n_rows, -1))[:,:2]
        s2_output = s2_output.reshape((n_rows, -1))[:,:2]
        
        s1_output_norm = np.sqrt(np.sum(s1_output * s1_output, axis=1))
        s2_output_norm = np.sqrt(np.sum(s2_output * s2_output, axis=1))
        
        denom = s1_output_norm * s2_output_norm
        
        if denom.any() == 0:
            raise Exception("Division by Zero!")
        s1_dot_s2 = np.sum(s1_output[:,:2] * s2_output[:,:2], axis=1) / denom
        
        complement = np.ones_like(s1_dot_s2) - s1_dot_s2
        out = np.array([s1_dot_s2,
                        complement]).T

        return out

short_lists = [["TBH", "to be honest"], ["FYI", "for your information"], ["BRB", "be right back"]]
# Function for replacing low occuring word(s) with <unk> token
def replace(box):
    if isinstance(box, Word) and dataset.count(box.name) < 2:
        return Word('unk', box.cod, box.dom)
    return box

def preprocess_df(df, ansatz):
    # Create raw diagram for both datasets.
    # We require both sentences to have a diagram
    # so it can be part of the dataset.
    # initializing the replace
    
    #replace_functor = Functor(ob=lambda x: x, ar=replace)

    rewriter = Rewriter(
        ['prepositional_phrase', 'determiner', 'auxiliary', 'curry', 'coordination', 'connector', 'preadverb', 'postadverb',
        'prepositional_phrase'])
    #rewriter = Rewriter(['curry', 'prepositional_phrase', 'connector'])
    parser = spiders_reader
    #parser = BobcatParser(verbose='text', root_cats=['S[dcl]', 'S[wq]', 'S[q]', 'S[qem]'])
    df["raw_s1_diagrams"] = parser.sentences2diagrams(list(df["s1"].values))
    df["raw_s2_diagrams"] = parser.sentences2diagrams(list(df["s2"].values))
      
    # Apply rewriter
    df["raw_s1_diagrams"] = df["raw_s1_diagrams"].apply(lambda d: rewriter(d))
    df["raw_s2_diagrams"] = df["raw_s2_diagrams"].apply(lambda d: rewriter(d))
      
    # Convert to normal form
    df["s1_diagrams"] = df["raw_s1_diagrams"].apply(lambda d: d.normal_form())
    df["s2_diagrams"] = df["raw_s2_diagrams"].apply(lambda d: d.normal_form())

    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    # Create circuits
    df["s1_circuit"] = df["s1_diagrams"].apply(lambda d: ansatz(remove_cups(d)))
    df["s2_circuit"] = df["s2_diagrams"].apply(lambda d: ansatz(remove_cups(d)))
def preprocess_df_old(df, ansatz):
    # Create raw diagram for both datasets.
    # We require both sentences to have a diagram
    # so it can be part of the dataset.
    parser = BobcatParser()
    df["s1_diagrams"] = parser.sentences2diagrams(list(df["s1"].values), suppress_exceptions=True)
    df["s2_diagrams"] = parser.sentences2diagrams(list(df["s2"].values), suppress_exceptions=True)
    df.dropna(inplace=True)

    # Convert to normal form
    df["s1_diagrams"] = df["s1_diagrams"].apply(lambda d: d.normal_form())
    df["s2_diagrams"] = df["s2_diagrams"].apply(lambda d: d.normal_form())

    # Vectorize label
    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])

    # Create circuits
    df["s1_circuit"] = df["s1_diagrams"].apply(lambda d: ansatz(remove_cups(d)))
    df["s2_circuit"] = df["s2_diagrams"].apply(lambda d: ansatz(remove_cups(d)))
    
def preprocess_df_norewrite(df, ansatz):
    # Create raw diagram for both datasets.
    # We require both sentences to have a diagram
    # so it can be part of the dataset.
    # initializing the replace
    
    parser = BobcatParser(verbose='text')
    df["s1_diagrams"] = parser.sentences2diagrams(list(df["s1"].values), suppress_exceptions=True)
    df["s2_diagrams"] = parser.sentences2diagrams(list(df["s2"].values), suppress_exceptions=True)
      
    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    # Create circuits
    df["s1_circuit"] = df["s1_diagrams"].apply(lambda d: ansatz(remove_cups(d.normal_form())))
    df["s2_circuit"] = df["s2_diagrams"].apply(lambda d: ansatz(remove_cups(d.normal_form())))


def preprocess_tensor(df, ansatz):
    
    parser = BobcatParser(verbose='text')
    df["s1_diagram"] = parser.sentences2diagrams(list(df["s1"].values), suppress_exceptions=True)
    df["s2_diagram"] = parser.sentences2diagrams(list(df["s2"].values), suppress_exceptions=True)
    
    # Convert to normal form
    df["s1_diagram"] = df["s1_diagram"].apply(lambda d: d.normal_form())
    df["s2_diagram"] = df["s2_diagram"].apply(lambda d: d.normal_form())
    
    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    
    df["s1_circuit"] = df["s1_diagram"].apply(lambda d: ansatz(d))
    df["s2_circuit"] = df["s2_diagram"].apply(lambda d: ansatz(d))
    
    
def preprocess_tensor_rewriter(df, ansatz):
    
    rewriter = Rewriter(
        ['prepositional_phrase', 'determiner', 'auxiliary', 'curry', 'coordination', 'connector', 'preadverb', 'postadverb',
        'prepositional_phrase'])
    parser = spiders_reader
    df["raw_s1_diagrams"] = parser.sentences2diagrams(list(df["s1"].values))
    df["raw_s2_diagrams"] = parser.sentences2diagrams(list(df["s2"].values))
      
    # Apply rewriter
    df["raw_s1_diagrams"] = df["raw_s1_diagrams"].apply(lambda d: rewriter(d))
    df["raw_s2_diagrams"] = df["raw_s2_diagrams"].apply(lambda d: rewriter(d))
      
    # Convert to normal form
    df["s1_diagrams"] = df["raw_s1_diagrams"].apply(lambda d: d.normal_form())
    df["s2_diagrams"] = df["raw_s2_diagrams"].apply(lambda d: d.normal_form())

    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    
    df["s1_circuit"] = df["s1_diagrams"].apply(lambda d: ansatz(d))
    df["s2_circuit"] = df["s2_diagrams"].apply(lambda d: ansatz(d))
    


def preprocess_custom(df, ansatz):
    
    parser = BobcatParser(verbose='text')
    df["s1_diagram"] = parser.sentences2diagrams(list(df["s1"].values), suppress_exceptions=True)
    df["s2_diagram"] = parser.sentences2diagrams(list(df["s2"].values), suppress_exceptions=True)
    
    df["label_v"] = df["label"].apply(lambda l: [0, 1] if l == 0 else [1, 0])
    
    
    df["s1_circuit"] = df["s1_diagram"].apply(lambda d: ansatz(d))
    df["s2_circuit"] = df["s2_diagram"].apply(lambda d: ansatz(d))
    
    
class CustomTketModel(TketModel):
    def forward(self, x: list[[Diagram, Diagram]]) -> np.ndarray:
        # The forward pass takes x with 2 circuits
        # for each of the sentence being compared
        s1_diagrams = []
        s2_diagrams = []
        n_rows = len(x)
        for s1d, s2d in x:
            s1_diagrams.append(s1d)
            s2_diagrams.append(s2d)
        
        s1_output = self.get_diagram_output(s1_diagrams)
        s2_output = self.get_diagram_output(s2_diagrams)
        s1_output = s1_output.reshape((n_rows, -1))[:,:2]
        s2_output = s2_output.reshape((n_rows, -1))[:,:2]
        
        s1_output_norm = np.sqrt(np.sum(s1_output * s1_output, axis=1))
        s2_output_norm = np.sqrt(np.sum(s2_output * s2_output, axis=1))
        denom = s1_output_norm * s2_output_norm
        s1_dot_s2 = np.sum(s1_output[:,:2] * s2_output[:,:2], axis=1) / denom

        complement = np.ones_like(s1_dot_s2) - s1_dot_s2
        out = np.array([s1_dot_s2,
                        complement]).T

        return out
