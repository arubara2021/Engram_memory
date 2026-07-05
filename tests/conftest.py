
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import tempfile
import textwrap
from typing import Generator, List

import pytest


@pytest.fixture
def tmp_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_text() -> str:
    return textwrap.dedent(
        """\
        Chapter 1: Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed.

        The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through trial and error.

        Chapter 2: Neural Networks

        A neural network is a computing system inspired by biological neural networks. It consists of layers of interconnected nodes called neurons. Each connection has a weight that adjusts during training.

        The input layer receives the data. Hidden layers process the information through mathematical transformations. The output layer produces the final prediction or classification.

        Deep learning refers to neural networks with multiple hidden layers. Convolutional neural networks are specialized for image processing. Recurrent neural networks handle sequential data like text and time series.

        Chapter 3: Training Process

        Training a neural network involves forward propagation and backpropagation. During forward propagation, input data flows through the network to produce an output. The loss function measures how far the output is from the expected result.

        Backpropagation calculates gradients of the loss with respect to each weight. Gradient descent uses these gradients to update the weights. The learning rate controls the size of each update step.

        Overfitting occurs when a model learns the training data too well and fails to generalize. Regularization techniques like dropout and weight decay help prevent overfitting.
        """
    )


@pytest.fixture
def sample_code() -> str:
    return textwrap.dedent(
        """\
        import os
        import sys
        from typing import List, Optional

        class UserService:
            def __init__(self, db_connection):
                self.db = db_connection
                self.cache = {}

            def get_user(self, user_id: int) -> Optional[dict]:
                if user_id in self.cache:
                    return self.cache[user_id]
                result = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
                if result:
                    self.cache[user_id] = result
                return result

            def create_user(self, name: str, email: str) -> dict:
                user = {"name": name, "email": email}
                return self.db.insert("users", user)

            def delete_user(self, user_id: int) -> bool:
                self.cache.pop(user_id, None)
                return self.db.delete("users", user_id)

        class OrderService:
            def __init__(self, db_connection):
                self.db = db_connection

            def get_order(self, order_id: int) -> Optional[dict]:
                return self.db.query(f"SELECT * FROM orders WHERE id = {order_id}")

            def create_order(self, user_id: int, items: List[dict]) -> dict:
                order = {"user_id": user_id, "items": items}
                return self.db.insert("orders", order)

        def main():
            db = connect_to_database()
            user_svc = UserService(db)
            order_svc = OrderService(db)

            user = user_svc.create_user("Alice", "alice@example.com")
            order = order_svc.create_order(user["id"], [{"item": "book", "qty": 1}])
            print(f"Created order {order['id']} for {user['name']}")

        if __name__ == "__main__":
            main()
        """
    )


@pytest.fixture
def sample_json() -> str:
    return textwrap.dedent(
        """\
        {
            "name": "Engram Project",
            "version": "1.0.0",
            "description": "A RAG engine for document processing",
            "features": [
                "hash-based retrieval",
                "vector embeddings",
                "dual retrieval",
                "concept extraction"
            ],
            "config": {
                "max_chunks": 150,
                "overlap_sentences": 2,
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "authors": [
                {"name": "Alice", "role": "lead"},
                {"name": "Bob", "role": "contributor"}
            ]
        }
        """
    )


@pytest.fixture
def sample_markdown() -> str:
    return textwrap.dedent(
        """\
        # Engram Documentation

        ## Overview

        Engram is a universal RAG engine that processes documents and enables fast retrieval.

        ## Features

        - **Hash-based retrieval**: O(1) keyword lookup using triple hashing
        - **Vector embeddings**: Semantic similarity via cosine distance
        - **Dual retrieval**: Combines hash and vector results
        - **Concept extraction**: TF-IDF based concept identification

        ## Installation

        Install using pip:

        ```bash
        pip install engram
        ```

        ## Usage

        Basic usage is simple:

        ```python
        from engram import Engram

        engine = Engram()
        engine.ingest("document.pdf")
        results = engine.retrieve("your question")
        ```

        ### Configuration

        Configure the engine using `EngramConfig`:

        ```python
        from engram import EngramConfig

        config = EngramConfig(
            chunk_strategy="recursive",
            embedding_enabled=True,
            retrieval_method="dual"
        )
        ```

        ## Architecture

        The system has three main layers:

        1. **Ingestion**: Parses and cleans documents
        2. **Indexing**: Builds hash tables and concept graphs
        3. **Retrieval**: Fast query processing with re-ranking
        """
    )


@pytest.fixture
def sample_csv() -> str:
    return textwrap.dedent(
        """\
        name,age,department,salary
        Alice,30,Engineering,95000
        Bob,25,Marketing,65000
        Charlie,35,Engineering,110000
        Diana,28,Design,75000
        Eve,32,Marketing,70000
        """
    )


@pytest.fixture
def sample_html() -> str:
    return textwrap.dedent(
        """\
        <!DOCTYPE html>
        <html>
        <head>
            <title>Engram Test Page</title>
            <style>body { font-family: sans-serif; }</style>
        </head>
        <body>
            <h1>Welcome to Engram</h1>
            <p>Engram is a powerful document processing engine.</p>
            <h2>Key Features</h2>
            <ul>
                <li>Fast hash-based retrieval</li>
                <li>Vector similarity search</li>
                <li>Multi-format support</li>
            </ul>
            <h2>Getting Started</h2>
            <p>Install the package and start processing documents immediately.</p>
            <script>console.log("This should be stripped");</script>
        </body>
        </html>
        """
    )


@pytest.fixture
def sample_yaml() -> str:
    return textwrap.dedent(
        """\
        project:
          name: engram
          version: 1.0.0
          description: RAG engine

        config:
          chunk_size: 150
          overlap: 2
          embedding_model: all-MiniLM-L6-v2

        features:
          - hash_retrieval
          - vector_search
          - dual_mode
          - concept_extraction
        """
    )


@pytest.fixture
def sample_toml() -> str:
    return textwrap.dedent(
        """\
        [project]
        name = "engram"
        version = "1.0.0"
        description = "RAG engine"

        [config]
        chunk_size = 150
        overlap = 2
        embedding_model = "all-MiniLM-L6-v2"

        [features]
        hash_retrieval = true
        vector_search = true
        dual_mode = false
        """
    )


@pytest.fixture
def write_file(tmp_dir: str):
    def _write(filename: str, content: str) -> str:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    return _write