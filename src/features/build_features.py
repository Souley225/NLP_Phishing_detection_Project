"""
Feature engineering pour la détection de phishing par analyse d'URLs.

Ce module implémente des techniques de NLP et d'extraction de features
spécifiques aux URLs pour la détection de sites de phishing.

Techniques implémentées (basées sur recherche SOTA):
1. TF-IDF mots: Capture les patterns de tokens dans l'URL
2. TF-IDF caractères: Détecte les obfuscations et typosquatting
3. Features lexicales: Longueur, entropie, structure de l'URL

Sources:
- Haynes et al. (2021): BERT pour URLs, detection mobile-friendly
- Rao et al. (2022): Features lexicales sans dépendances tierces
- Kalla & Kuraku (2023): TF-IDF + ML hybride

Auteur: Souleymane Sall
Email: sallsouleymane2207@gmail.com
"""

import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# Liste des TLDs suspects fréquemment utilisés pour le phishing
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  # TLDs gratuits populaires
    ".pw", ".cc", ".top", ".xyz", ".club",
    ".work", ".click", ".link", ".racing",
}


def extract_lexical_features(url: str) -> dict[str, float]:
    """
    Extrait les features lexicales d'une URL.
    
    Ces features capturent la structure et les caractéristiques statistiques
    de l'URL sans dépendre de services tiers (WHOIS, DNS, etc.).
    
    Features extraites:
    - url_length: Longueur totale de l'URL
    - num_dots: Nombre de points (indique la profondeur des sous-domaines)
    - num_slashes: Nombre de slashes (profondeur du chemin)
    - num_dashes: Tirets dans l'URL (souvent utilisés pour typosquatting)
    - num_underscores: Underscores (pattern suspect)
    - num_at: Symbole @ (utilisé pour masquer le vrai domaine)
    - num_question: Points d'interrogation (paramètres GET)
    - num_ampersand: Symboles & (multiples paramètres)
    - num_equal: Symboles = (paires clé-valeur)
    - num_digits: Nombre de chiffres
    - digit_ratio: Ratio chiffres/longueur totale
    - has_ip: Présence d'une adresse IP dans l'URL
    - entropy: Entropie de Shannon (mesure du désordre)
    - subdomain_count: Nombre de sous-domaines
    - path_depth: Profondeur du chemin
    - suspicious_tld: TLD dans la liste des TLDs suspects
    
    Args:
        url: URL à analyser
    
    Returns:
        Dictionnaire de features
    """
    features = {}
    
    # Feature 1: Longueur totale
    features["url_length"] = len(url)
    
    # Features 2-9: Comptage de caractères spéciaux
    features["num_dots"] = url.count(".")
    features["num_slashes"] = url.count("/")
    features["num_dashes"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_at"] = url.count("@")
    features["num_question"] = url.count("?")
    features["num_ampersand"] = url.count("&")
    features["num_equal"] = url.count("=")
    
    # Features 10-11: Analyse des chiffres
    num_digits = sum(c.isdigit() for c in url)
    features["num_digits"] = num_digits
    features["digit_ratio"] = num_digits / len(url) if len(url) > 0 else 0.0
    
    # Feature 12: Présence d'adresse IP
    # Détecte les patterns IPv4 (ex: 192.168.1.1)
    ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    features["has_ip"] = 1.0 if re.search(ip_pattern, url) else 0.0
    
    # Feature 13: Entropie de Shannon
    # Mesure le désordre/aléatoire dans l'URL
    # URLs légitimes ont généralement une entropie plus faible
    features["entropy"] = calculate_entropy(url)
    
    # Feature 14: Nombre de sous-domaines
    # Parse l'URL pour extraire le domaine
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc or parsed.path.split("/")[0]
        # Compter les points dans le hostname (hors TLD)
        parts = hostname.split(".")
        features["subdomain_count"] = max(0, len(parts) - 2)  # -2 pour domain + TLD
    except Exception:
        features["subdomain_count"] = 0
    
    # Feature 15: Profondeur du chemin
    # Nombre de segments dans le chemin après le domaine
    try:
        parsed = urlparse(url)
        path = parsed.path
        features["path_depth"] = len([p for p in path.split("/") if p])
    except Exception:
        features["path_depth"] = 0
    
    # Feature 16: TLD suspect
    # Vérifie si l'URL utilise un TLD connu pour le phishing
    url_lower = url.lower()
    features["suspicious_tld"] = 1.0 if any(tld in url_lower for tld in SUSPICIOUS_TLDS) else 0.0
    
    return features


def calculate_entropy(text: str) -> float:
    """
    Calcule l'entropie de Shannon d'une chaîne de caractères.
    
    L'entropie mesure le désordre/aléatoire dans le texte.
    Une entropie élevée indique un texte plus aléatoire (suspect).
    
    Formule: H(X) = -Σ p(x) * log2(p(x))
    
    Args:
        text: Texte à analyser
    
    Returns:
        Entropie de Shannon (bits)
    """
    if not text:
        return 0.0
    
    # Calculer la fréquence de chaque caractère
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculer les probabilités
    text_length = len(text)
    probabilities = [count / text_length for count in char_counts.values()]
    
    # Calculer l'entropie
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    return entropy


def tokenize_url(url: str) -> str:
    """
    Tokenise une URL en séparant par les délimiteurs communs.
    
    Remplace les délimiteurs (/, ., -, _, ?, &, =) par des espaces
    pour créer des tokens utilisables par TF-IDF.
    
    Exemples:
        "http://example.com/path" -> "http example com path"
        "user-login.php?id=123" -> "user login php id 123"
    
    Args:
        url: URL à tokeniser
    
    Returns:
        URL tokenisée (tokens séparés par des espaces)
    """
    # Remplacer les délimiteurs par des espaces
    tokens = re.sub(r"[/.\-_?&=:@]", " ", url)
    # Nettoyer les espaces multiples
    tokens = re.sub(r"\s+", " ", tokens)
    return tokens.strip().lower()


class URLFeatureExtractor:
    """
    Classe pour extraire et transformer les features des URLs.
    
    Cette classe combine:
    - TF-IDF sur les mots (tokens)
    - TF-IDF sur les caractères (n-grams)
    - Features lexicales numériques
    
    Attributes:
        config: Configuration Hydra des features
        tfidf_word: Vectorizer TF-IDF pour les mots
        tfidf_char: Vectorizer TF-IDF pour les caractères
        scaler: StandardScaler pour les features numériques
        is_fitted: Indicateur si le transformer est entraîné
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialise le feature extractor.
        
        Args:
            config: Configuration des features depuis Hydra
        """
        self.config = config
        self.tfidf_word: TfidfVectorizer | None = None
        self.tfidf_char: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.is_fitted = False
        
        # Initialiser les vectorizers selon la config
        self._init_vectorizers()
    
    def _init_vectorizers(self) -> None:
        """Initialise les vectorizers TF-IDF selon la configuration."""
        # TF-IDF mots
        if self.config["tfidf_word"]["use"]:
            self.tfidf_word = TfidfVectorizer(
                ngram_range=tuple(self.config["tfidf_word"]["ngram_range"]),
                max_features=self.config["tfidf_word"]["max_features"],
                min_df=self.config["tfidf_word"]["min_df"],
                max_df=self.config["tfidf_word"]["max_df"],
                tokenizer=lambda x: x.split(),  # Déjà tokenisé
                lowercase=True,
            )
        
        # TF-IDF caractères
        if self.config["tfidf_char"]["use"]:
            self.tfidf_char = TfidfVectorizer(
                analyzer=self.config["tfidf_char"]["analyzer"],
                ngram_range=tuple(self.config["tfidf_char"]["ngram_range"]),
                max_features=self.config["tfidf_char"]["max_features"],
                min_df=self.config["tfidf_char"]["min_df"],
                max_df=self.config["tfidf_char"]["max_df"],
                lowercase=True,
            )
        
        # Scaler pour les features numériques
        if self.config["lexical"]["use"]:
            self.scaler = StandardScaler()
    
    def fit(self, urls: pd.Series) -> "URLFeatureExtractor":
        """
        Entraîne les vectorizers et le scaler sur les URLs.
        
        Args:
            urls: Série pandas d'URLs
        
        Returns:
            self (pour chaînage)
        """
        # TF-IDF mots
        if self.tfidf_word is not None:
            tokenized_urls = urls.apply(tokenize_url)
            self.tfidf_word.fit(tokenized_urls)
        
        # TF-IDF caractères
        if self.tfidf_char is not None:
            self.tfidf_char.fit(urls)
        
        # Scaler pour features lexicales
        if self.scaler is not None:
            lexical_features = self._extract_lexical_features_batch(urls)
            self.scaler.fit(lexical_features)
        
        self.is_fitted = True
        return self
    
    def transform(self, urls: pd.Series) -> np.ndarray:
        """
        Transforme les URLs en features.
        
        Args:
            urls: Série pandas d'URLs
        
        Returns:
            Matrice de features (array 2D)
        
        Raises:
            ValueError: Si le transformer n'est pas entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le transformer doit être entraîné avec fit() avant transform()")
        
        features_list = []
        
        # Features TF-IDF mots
        if self.tfidf_word is not None:
            tokenized_urls = urls.apply(tokenize_url)
            tfidf_word_features = self.tfidf_word.transform(tokenized_urls).toarray()
            features_list.append(tfidf_word_features)
        
        # Features TF-IDF caractères
        if self.tfidf_char is not None:
            tfidf_char_features = self.tfidf_char.transform(urls).toarray()
            features_list.append(tfidf_char_features)
        
        # Features lexicales
        if self.scaler is not None:
            lexical_features = self._extract_lexical_features_batch(urls)
            lexical_features_scaled = self.scaler.transform(lexical_features)
            features_list.append(lexical_features_scaled)
        
        # Concaténer toutes les features
        X = np.hstack(features_list) if features_list else np.array([]).reshape(len(urls), 0)
        
        return X
    
    def fit_transform(self, urls: pd.Series) -> np.ndarray:
        """
        Entraîne et transforme les URLs en une seule étape.
        
        Args:
            urls: Série pandas d'URLs
        
        Returns:
            Matrice de features (array 2D)
        """
        self.fit(urls)
        return self.transform(urls)
    
    def _extract_lexical_features_batch(self, urls: pd.Series) -> np.ndarray:
        """
        Extrait les features lexicales pour un batch d'URLs.
        
        Args:
            urls: Série pandas d'URLs
        
        Returns:
            Matrice de features lexicales (array 2D)
        """
        features_list = []
        
        for url in urls:
            features = extract_lexical_features(str(url))
            features_list.append(list(features.values()))
        
        return np.array(features_list)
    
    def get_feature_names(self) -> list[str]:
        """
        Retourne les noms de toutes les features.
        
        Returns:
            Liste des noms de features
        """
        feature_names = []
        
        # Noms des features TF-IDF mots
        if self.tfidf_word is not None:
            word_features = [f"tfidf_word_{name}" for name in self.tfidf_word.get_feature_names_out()]
            feature_names.extend(word_features)
        
        # Noms des features TF-IDF caractères
        if self.tfidf_char is not None:
            char_features = [f"tfidf_char_{name}" for name in self.tfidf_char.get_feature_names_out()]
            feature_names.extend(char_features)
        
        # Noms des features lexicales
        if self.scaler is not None:
            lexical_names = list(self.config["lexical"]["features"])
            feature_names.extend(lexical_names)
        
        return feature_names