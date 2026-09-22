from backend.app.core.config import settings


class Neo4jService:

    def __init__(self) -> None:

        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Neo4j support requires the 'neo4j' package. "
                "Install backend/requirements.txt before building a graph."
            ) from exc

        if not settings.NEO4J_URI:
            raise RuntimeError(
                "NEO4J_URI is not configured."
            )

        if not settings.NEO4J_PASSWORD:
            raise RuntimeError(
                "NEO4J_PASSWORD is not configured."
            )

        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD,
            ),
        )

    def verify_connection(self) -> bool:

        self.driver.verify_connectivity()

        return True

    def create_graph(
        self,
        document_name: str,
        page: int,
        chunk_id: str,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:

        with self.driver.session() as session:

            session.execute_write(
                self._create_graph_transaction,
                document_name,
                page,
                chunk_id,
                entities,
                relationships,
            )

    @staticmethod
    def _create_graph_transaction(
        tx,
        document_name,
        page,
        chunk_id,
        entities,
        relationships,
    ):

        # Create source chunk.

        tx.run(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.page = $page,
                c.document = $document_name
            """,
            chunk_id=chunk_id,
            page=page,
            document_name=document_name,
        )

        # Create entities.

        for entity in entities:

            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "Concept").strip()

            if not name:
                continue

            safe_type = "".join(
                character
                for character in entity_type
                if character.isalnum()
            )

            if not safe_type:
                safe_type = "Concept"

            query = f"""
            MERGE (e:Entity {{name: $name, type: $type}})
            MERGE (c:Chunk {{id: $chunk_id}})
            MERGE (c)-[:MENTIONS]->(e)
            """

            tx.run(
                query,
                name=name,
                type=safe_type,
                chunk_id=chunk_id,
            )

        # Create relationships.

        for relationship in relationships:

            source = relationship.get(
                "source",
                "",
            ).strip()

            target = relationship.get(
                "target",
                "",
            ).strip()

            relation = relationship.get(
                "relation",
                "RELATED_TO",
            ).strip()

            if not source or not target:
                continue

            safe_relation = "".join(
                character
                for character in relation
                if character.isalnum()
                or character == "_"
            )

            if not safe_relation:
                safe_relation = "RELATED_TO"

            query = f"""
            MERGE (source:Entity {{name: $source}})
            MERGE (target:Entity {{name: $target}})
            MERGE (source)-[:{safe_relation}]->(target)
            """

            tx.run(
                query,
                source=source,
                target=target,
            )

    def get_graph_stats(self) -> dict:

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (n)
                RETURN count(n) AS nodes
                """
            )

            nodes = result.single()["nodes"]

            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS relationships
                """
            )

            relationships = result.single()[
                "relationships"
            ]

        return {
            "nodes": nodes,
            "relationships": relationships,
        }

    def close(self) -> None:

        self.driver.close()