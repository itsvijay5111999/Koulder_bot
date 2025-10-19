"""
RAG Knowledge Base for Research Papers
Stores and retrieves research papers using ChromaDB and Groq LLM
"""

import os
import json
import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import List, Dict, Optional

# Vector DB and Embeddings
import chromadb
from chromadb.utils import embedding_functions

# Groq for LLM
from groq import Groq

# Configuration
CHROMA_DB_PATH = "./chroma_papers_db"
COLLECTION_NAME = "research_papers"
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "llama3-70b-8192"

class ResearchPaperRAG:
    """RAG system for research papers using ChromaDB and Groq"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the RAG system"""
        self.groq_api_key = groq_api_key
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Use default embedding function (all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            print(f"✅ Loaded existing collection with {self.collection.count()} papers")
        except:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"description": "AI/ML Research Papers from arXiv"}
            )
            print("✅ Created new collection")
    
    def fetch_arxiv_papers(self, category: str = "cs.AI", max_results: int = 50) -> List[Dict]:
        """Fetch papers from arXiv API"""
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"cat:{category}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        query_url = base_url + "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        
        try:
            response = requests.get(query_url, timeout=30)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                papers = []
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    paper = {
                        'id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
                        'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' '),
                        'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' '),
                        'authors': [author.find('{http://www.w3.org/2005/Atom}name').text 
                                   for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                        'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                        'categories': [],
                        'arxiv_url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                    }
                    
                    # Get categories
                    for cat in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                        paper['categories'].append(cat.get('term'))
                    
                    papers.append(paper)
                
                return papers
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []
    
    def add_papers_to_db(self, papers: List[Dict]) -> int:
        """Add papers to ChromaDB"""
        if not papers:
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for paper in papers:
            paper_id = paper['id']
            
            # Check if paper already exists
            try:
                existing = self.collection.get(ids=[paper_id])
                if existing['ids']:
                    continue  # Skip if already exists
            except:
                pass
            
            # Create document text (title + summary for better retrieval)
            doc_text = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
            
            # Metadata
            metadata = {
                'title': paper['title'][:500],  # Limit length
                'authors': ', '.join(paper['authors'][:5]),
                'published': paper['published'][:10],
                'categories': ', '.join(paper['categories']),
                'arxiv_url': paper['arxiv_url'],
                'added_date': datetime.datetime.now().strftime("%Y-%m-%d")
            }
            
            ids.append(paper_id)
            documents.append(doc_text)
            metadatas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ Added {len(ids)} new papers to database")
            return len(ids)
        
        return 0
    
    def search_papers(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant papers using semantic search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['ids'] or not results['ids'][0]:
                return {
                    'success': False,
                    'message': 'No relevant papers found',
                    'papers': []
                }
            
            papers = []
            for i, paper_id in enumerate(results['ids'][0]):
                paper_data = {
                    'id': paper_id,
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'authors': results['metadatas'][0][i].get('authors', 'Unknown'),
                    'published': results['metadatas'][0][i].get('published', 'Unknown'),
                    'categories': results['metadatas'][0][i].get('categories', 'Unknown'),
                    'arxiv_url': results['metadatas'][0][i].get('arxiv_url', '#'),
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                papers.append(paper_data)
            
            return {
                'success': True,
                'papers': papers,
                'count': len(papers)
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Search error: {str(e)}',
                'papers': []
            }
    
    def answer_question(self, question: str, n_results: int = 5) -> Dict:
        """Answer a question using RAG"""
        # Search for relevant papers
        search_results = self.search_papers(question, n_results)
        
        if not search_results['success'] or not search_results['papers']:
            return {
                'success': False,
                'answer': 'I could not find any relevant research papers to answer your question. Try updating the database with latest papers.',
                'sources': []
            }
        
        # Prepare context from retrieved papers
        context_parts = []
        sources = []
        
        for i, paper in enumerate(search_results['papers'], 1):
            context_parts.append(f"Paper {i}:\n{paper['content']}\n")
            sources.append({
                'title': paper['title'],
                'authors': paper['authors'],
                'url': paper['arxiv_url'],
                'published': paper['published']
            })
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt for Groq
        prompt = f"""You are an AI research assistant. Answer the following question based on the research papers provided below.

Question: {question}

Research Papers Context:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the papers above
- Reference specific papers when making claims (e.g., "According to Paper 1...")
- If the papers don't fully answer the question, say so
- Be accurate and cite the papers appropriately
- Keep your answer concise but informative

Answer:"""
        
        try:
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI research assistant that answers questions based on academic papers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'papers_used': len(sources)
            }
        
        except Exception as e:
            return {
                'success': False,
                'answer': f'Error generating answer: {str(e)}',
                'sources': sources
            }
    
    def update_daily_papers(self, categories: List[str] = None) -> Dict:
        """Update database with latest papers from specified categories"""
        if categories is None:
            categories = ["cs.AI", "cs.LG", "cs.CL"]
        
        total_added = 0
        results = {}
        
        for category in categories:
            print(f"Fetching papers from {category}...")
            papers = self.fetch_arxiv_papers(category=category, max_results=30)
            added = self.add_papers_to_db(papers)
            total_added += added
            results[category] = {
                'fetched': len(papers),
                'added': added
            }
        
        return {
            'success': True,
            'total_added': total_added,
            'categories': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                'success': True,
                'total_papers': count,
                'collection_name': COLLECTION_NAME,
                'db_path': CHROMA_DB_PATH
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_database(self):
        """Clear all papers from the database"""
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            return {'success': True, 'message': 'Database cleared successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Convenience functions for easy import
def initialize_rag(groq_api_key: str) -> ResearchPaperRAG:
    """Initialize the RAG system"""
    return ResearchPaperRAG(groq_api_key)


def update_papers(rag_system: ResearchPaperRAG, categories: List[str] = None):
    """Update papers in the database"""
    return rag_system.update_daily_papers(categories)


def ask_question(rag_system: ResearchPaperRAG, question: str, n_results: int = 5):
    """Ask a question to the RAG system"""
    return rag_system.answer_question(question, n_results)


# Example usage
if __name__ == "__main__":
    # Initialize with your Groq API key
    GROQ_API_KEY = "your_groq_api_key_here"
    
    rag = ResearchPaperRAG(GROQ_API_KEY)
    
    # Update database with latest papers
    print("Updating database with latest papers...")
    update_result = rag.update_daily_papers(categories=["cs.AI", "cs.LG", "cs.CL"])
    print(f"Added {update_result['total_added']} new papers")
    
    # Get stats
    stats = rag.get_stats()
    print(f"Database has {stats['total_papers']} papers")
    
    # Ask a question
    question = "What are the latest developments in large language models?"
    print(f"\nQuestion: {question}")
    result = rag.answer_question(question)
    
    if result['success']:
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nBased on {result['papers_used']} papers")
    else:
        print(f"Error: {result['answer']}")