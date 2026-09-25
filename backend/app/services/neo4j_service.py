import truststore

# Use the Windows native certificate store.
# This helps Python trust certificates installed by Windows,
# antivirus software, VPNs, proxies, or enterprise environments.
truststore.inject_into_ssl()

from neo4j import GraphDatabase
from backend.app.core.config import settings


class Neo4jService:
    """
    Service responsible for:
    1. Connecting to Neo4j Aura
    2. Creating the knowledge graph
    3. Retrieving graph information
    4. Returning graph statistics
    """

    def __init__(self) -> None:
        if not settings.NEO4J_URI:
            raise RuntimeError("NEO4J_URI is not configured.")

        if not settings.NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD is not configured.")

        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD,
            ),
        )

    # ---------------------------------------------------------
    # CONNECTION TEST
    # ---------------------------------------------------------

    def verify_connection(self) -> bool:
        """
        Verify that Neo4j Aura is reachable.
        """
        self.driver.verify_connectivity()
        return True

    # ---------------------------------------------------------
    # GRAPH CREATION
    # ---------------------------------------------------------

    def create_graph(
        self,
        document_name: str,
        document_id: str,
        page: int,
        chunk_id: str,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:
        """
        Create a Chunk node, Entity nodes and relationships
        extracted from a document chunk.
        """

        with self.driver.session() as session:
            session.execute_write(
                self._create_graph_transaction,
                document_name,
                document_id,
                page,
                chunk_id,
                entities,
                relationships,
            )

    @staticmethod
    def _create_graph_transaction(
        tx,
        document_name: str,
        document_id: str,
        page: int,
        chunk_id: str,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:

        # -----------------------------------------------------
        # CREATE CHUNK NODE
        # -----------------------------------------------------

        tx.run(
            """
            MERGE (c:Chunk {id: $scoped_chunk_id})
            SET
                c.page = $page,
                c.document = $document_name,
                c.document_id = $document_id,
                c.chunk_id = $chunk_id
            """,
            scoped_chunk_id=f"{document_id}:{chunk_id}",
            chunk_id=chunk_id,
            page=page,
            document_name=document_name,
            document_id=document_id,
        )

        # -----------------------------------------------------
        # CREATE ENTITY NODES
        # -----------------------------------------------------

        for entity in entities:

            name = str(entity.get("name", "")).strip()

            entity_type = str(
                entity.get("type", "Concept")
            ).strip()

            if not name:
                continue

            # Neo4j relationship/node labels cannot safely be
            # passed as parameters, so sanitize the type.
            safe_type = "".join(
                character
                for character in entity_type
                if character.isalnum()
            )

            if not safe_type:
                safe_type = "Concept"

            tx.run(
                f"""
                MERGE (e:Entity:{safe_type} {{document_id: $document_id, name: $name}})
                SET e.type = $type

                MERGE (c:Chunk {{id: $scoped_chunk_id}})
                MERGE (c)-[:MENTIONS {{document_id: $document_id}}]->(e)
                """,
                name=name,
                document_id=document_id,
                type=safe_type,
                scoped_chunk_id=f"{document_id}:{chunk_id}",
            )

        # -----------------------------------------------------
        # CREATE ENTITY RELATIONSHIPS
        # -----------------------------------------------------

        for relationship in relationships:

            source = str(
                relationship.get("source", "")
            ).strip()

            target = str(
                relationship.get("target", "")
            ).strip()

            relation = str(
                relationship.get("relation", "RELATED_TO")
            ).strip()

            if not source or not target:
                continue

            # Relationship types must be inserted into the
            # Cypher query itself, so sanitize them first.
            safe_relation = "".join(
                character
                for character in relation
                if character.isalnum() or character == "_"
            )

            if not safe_relation:
                safe_relation = "RELATED_TO"

            tx.run(
                f"""
                MERGE (source:Entity {{document_id: $document_id, name: $source}})
                MERGE (target:Entity {{document_id: $document_id, name: $target}})
                MERGE (source)-[:{safe_relation} {{document_id: $document_id}}]->(target)
                """,
                document_id=document_id,
                source=source,
                target=target,
            )

    # ---------------------------------------------------------
    # GRAPH RETRIEVAL
    # ---------------------------------------------------------

    def search_graph(
        self,
        query: str,
        document_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search the Neo4j knowledge graph using words from the
        user's query.

        For matching entities, return:
        - entity name
        - entity type
        - connected entities
        - relationship types

        This is the first GraphRAG retrieval layer.
        """

        # Normalize query into searchable words.
        words = [
            word.strip().lower()
            for word in query.split()
            if len(word.strip()) >= 3
        ]

        if not words:
            return []

        # Prevent unreasonable limits.
        limit = max(1, min(limit, 50))

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (e:Entity {document_id: $document_id})

                WHERE any(
                    word IN $words
                    WHERE toLower(e.name) CONTAINS word
                )

                OPTIONAL MATCH (e)-[r]-(connected:Entity {document_id: $document_id})

                RETURN
                    e.name AS entity,
                    e.type AS entity_type,

                    collect(
                        DISTINCT {
                            relationship: type(r),
                            connected_entity: connected.name,
                            connected_type: connected.type
                        }
                    )[0..10] AS relationships

                LIMIT $limit
                """,
                words=words,
                document_id=document_id,
                limit=limit,
            )

            results = []

            for record in result:

                relationships = record["relationships"] or []

                # Remove empty relationship records.
                clean_relationships = [
                    relationship
                    for relationship in relationships
                    if relationship.get("connected_entity")
                ]

                results.append(
                    {
                        "entity": record["entity"],
                        "entity_type": record["entity_type"],
                        "document_id": document_id,
                        "relationships": clean_relationships,
                    }
                )

            return results

    # ---------------------------------------------------------
    # GRAPH STATISTICS
    # ---------------------------------------------------------

    def get_graph_stats(self) -> dict:
        """
        Return the number of nodes and relationships
        currently stored in Neo4j.
        """

        with self.driver.session() as session:

            node_result = session.run(
                """
                MATCH (n)
                RETURN count(n) AS nodes
                """
            )

            nodes = node_result.single()["nodes"]

            relationship_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS relationships
                """
            )

            relationships = relationship_result.single()[
                "relationships"
            ]

        return {
            "nodes": nodes,
            "relationships": relationships,
        }

    # ---------------------------------------------------------
    # CLOSE CONNECTION
    # ---------------------------------------------------------

    def close(self) -> None:
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()
