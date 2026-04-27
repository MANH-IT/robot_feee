"""
crawl_utc_advanced.py - Research-grade crawler for UTC website
Version: 2.1 (Fixed networkx issue)
"""

import os
import sys
import json
import time
import logging
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional, Tuple
from collections import deque

import requests
from bs4 import BeautifulSoup
import numpy as np

# ========== CONFIGURATION ==========
BASE_URL = "https://utc.edu.vn"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Tạo thư mục
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Created directory: {OUTPUT_DIR}")
except Exception as e:
    print(f"[ERROR] Cannot create directory: {e}")
    sys.exit(1)

CRAWL_DEPTH = 2
MAX_PAGES = 100
DELAY_BETWEEN_REQUESTS = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2

# Logging
LOG_FILE = os.path.join(OUTPUT_DIR, "crawler.log")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ========== DATA SCHEMA ==========
class UTCKnowledgeBase:
    def __init__(self):
        self.metadata = {
            "source": BASE_URL,
            "crawled_date": datetime.now().isoformat(),
            "version": "2.1",
            "total_pages": 0,
            "total_lecturers": 0,
            "total_departments": 0,
            "total_research": 0
        }
        self.departments: List[Dict] = []
        self.lecturers: List[Dict] = []
        self.research_projects: List[Dict] = []
        self.publications: List[Dict] = []
        self.labs: List[Dict] = []
        self.news: List[Dict] = []
        self.admissions: List[Dict] = []
        self.partnerships: List[Dict] = []
        self.pages: Dict[str, Dict] = {}
        self.qa_pairs: List[Dict] = []
        self.crawled_urls: Set[str] = set()
    
    def to_dict(self) -> Dict:
        return {
            "metadata": self.metadata,
            "departments": self.departments,
            "lecturers": self.lecturers,
            "research_projects": self.research_projects,
            "publications": self.publications,
            "labs": self.labs,
            "news": self.news,
            "admissions": self.admissions,
            "partnerships": self.partnerships,
            "qa_pairs": self.qa_pairs,
            "pages": self.pages
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved knowledge base to {filepath}")


# ========== TEXT CLEANING ==========
class TextCleaner:
    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\sàáảãạâầấẩẫậăằắẳẵặđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ,]', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()


# ========== CRAWLER ==========
class AdvancedCrawler:
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 100):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited: Set[str] = set()
        self.to_visit: deque = deque()
        self.kb = UTCKnowledgeBase()
        self.cleaner = TextCleaner()
        self.to_visit.append((base_url, 0))
    
    def _get_with_retry(self, url: str) -> Optional[requests.Response]:
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return None
        return None
    
    def _extract_internal_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == self.base_domain:
                full_url = parsed._replace(fragment='').geturl()
                if full_url not in self.visited and full_url != current_url:
                    links.append(full_url)
        return links
    
    def _extract_departments(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        departments = []
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            text = heading.get_text(strip=True)
            if any(kw in text.lower() for kw in ['khoa', 'viện', 'bộ môn', 'trung tâm']):
                desc = ""
                next_p = heading.find_next('p')
                if next_p:
                    desc = next_p.get_text(strip=True)[:500]
                departments.append({
                    "name": text[:100],
                    "description": self.cleaner.normalize(desc),
                    "url": url,
                    "faculty_members": []
                })
        return departments
    
    def _extract_lecturers(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        lecturers = []
        for name_tag in soup.find_all(['h3', 'h4', 'strong']):
            name = name_tag.get_text(strip=True)
            if 3 < len(name) < 50 and not any(x in name for x in ['khoa', 'viện', 'trung tâm', 'thông báo', 'tuyển sinh']):
                title = ""
                research = ""
                parent = name_tag.find_parent(['div', 'li', 'article'])
                if parent:
                    parent_text = parent.get_text(strip=True)
                    if any(x in parent_text for x in ['TS.', 'PGS.', 'ThS.', 'GS.']):
                        title_match = re.search(r'(PGS\.TS\.|TS\.|ThS\.|GS\.)[^.]*\.', parent_text)
                        if title_match:
                            title = title_match.group()
                    if 'nghiên cứu' in parent_text:
                        research_match = re.search(r'nghiên cứu[^.]*\.', parent_text)
                        if research_match:
                            research = research_match.group()
                lecturers.append({
                    "name": name[:50],
                    "title": self.cleaner.normalize(title)[:100],
                    "research_area": self.cleaner.normalize(research)[:200],
                    "department": None,
                    "url": url
                })
        return lecturers
    
    def _extract_research(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        research_items = []
        for article in soup.find_all(['article', 'li', 'div']):
            title_tag = article.find(['h3', 'h4', 'strong'])
            if title_tag:
                title = title_tag.get_text(strip=True)
                if any(kw in title.lower() for kw in ['nghiên cứu', 'đề tài', 'dự án', 'research', 'project']):
                    desc = ""
                    desc_tag = article.find('p')
                    if desc_tag:
                        desc = desc_tag.get_text(strip=True)[:500]
                    year = None
                    year_match = re.search(r'20\d{2}', title + desc)
                    if year_match:
                        year = year_match.group()
                    research_items.append({
                        "title": self.cleaner.normalize(title)[:200],
                        "description": self.cleaner.normalize(desc),
                        "year": year,
                        "url": url
                    })
        return research_items
    
    def _extract_news(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        news_items = []
        for article in soup.find_all(['article', 'li']):
            link_tag = article.find('a')
            if link_tag and link_tag.get('href'):
                title = link_tag.get_text(strip=True)
                if title and len(title) > 5 and len(title) < 200:
                    link = urljoin(url, link_tag['href'])
                    date = None
                    date_tag = article.find(class_=re.compile(r'date|time|ngay'))
                    if date_tag:
                        date = date_tag.get_text(strip=True)[:20]
                    news_items.append({
                        "title": self.cleaner.normalize(title),
                        "url": link,
                        "date": date,
                        "summary": ""
                    })
        return news_items
    
    def crawl(self):
        logger.info(f"Starting BFS crawl of {self.base_url}")
        
        while self.to_visit and len(self.visited) < self.max_pages:
            url, depth = self.to_visit.popleft()
            if url in self.visited:
                continue
            
            logger.info(f"Crawling [{depth}] {url[:80]}...")
            response = self._get_with_retry(url)
            if not response:
                self.visited.add(url)
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            page_data = {
                "url": url,
                "title": soup.find('title').get_text(strip=True)[:200] if soup.find('title') else "",
                "crawled_at": datetime.now().isoformat(),
                "depth": depth
            }
            self.kb.pages[url] = page_data
            
            # Extract entities
            for dept in self._extract_departments(soup, url):
                if dept not in self.kb.departments:
                    self.kb.departments.append(dept)
            
            for lec in self._extract_lecturers(soup, url):
                if lec not in self.kb.lecturers:
                    self.kb.lecturers.append(lec)
            
            for res in self._extract_research(soup, url):
                if res not in self.kb.research_projects:
                    self.kb.research_projects.append(res)
            
            for item in self._extract_news(soup, url):
                if item not in self.kb.news:
                    self.kb.news.append(item)
            
            if 'tuyen-sinh' in url or 'tuyển sinh' in response.text.lower():
                self.kb.admissions.extend(self._extract_news(soup, url))
            
            # BFS: thêm link mới
            if depth < self.max_depth:
                for link in self._extract_internal_links(soup, url):
                    if link not in self.visited and link not in [item[0] for item in self.to_visit]:
                        self.to_visit.append((link, depth + 1))
            
            self.visited.add(url)
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Update metadata
        self.kb.metadata["total_pages"] = len(self.kb.pages)
        self.kb.metadata["total_departments"] = len(self.kb.departments)
        self.kb.metadata["total_lecturers"] = len(self.kb.lecturers)
        self.kb.metadata["total_research"] = len(self.kb.research_projects)
        
        logger.info(f"Crawl completed. Pages: {len(self.kb.pages)}")
        return self.kb


# ========== Q&A GENERATOR ==========
class QAGenerator:
    @staticmethod
    def generate(kb: UTCKnowledgeBase) -> List[Dict]:
        qa_pairs = []
        
        qa_pairs.append({
            "question": "Trường Đại học Giao thông Vận tải thành lập năm nào?",
            "answer": "Trường Đại học Giao thông Vận tải được thành lập năm 1962.",
            "keywords": ["năm thành lập", "1962", "thành lập"]
        })
        
        for dept in kb.departments[:10]:
            if dept.get('name'):
                qa_pairs.append({
                    "question": f"Khoa {dept['name']} có những ngành gì?",
                    "answer": f"Thông tin về Khoa {dept['name']}: {dept.get('description', 'Chưa có mô tả')[:200]}",
                    "keywords": [dept['name'].lower(), "khoa"]
                })
        
        for lecturer in kb.lecturers[:20]:
            if lecturer.get('name'):
                qa_pairs.append({
                    "question": f"Thông tin về giảng viên {lecturer['name']}",
                    "answer": f"Giảng viên {lecturer['name']}. {lecturer.get('title', '')} Lĩnh vực: {lecturer.get('research_area', 'Chưa rõ')}",
                    "keywords": [lecturer['name'].lower()]
                })
        
        return qa_pairs


# ========== KNOWLEDGE GRAPH ==========
class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = None
        try:
            import networkx as nx
            self.nx = nx
            logger.info("NetworkX loaded")
        except ImportError:
            logger.warning("networkx not installed")
            self.nx = None
    
    def build(self, kb: UTCKnowledgeBase):
        if not self.nx:
            logger.warning("Skipping knowledge graph (networkx not available)")
            return None
        
        try:
            G = self.nx.DiGraph()
            
            # Thêm departments
            for dept in kb.departments:
                dept_name = dept.get('name', 'Unknown')
                if dept_name and dept_name != 'Unknown':
                    G.add_node(f"DEPT_{dept_name}", type="department", name=dept_name)
            
            # Thêm lecturers
            for lec in kb.lecturers:
                lec_name = lec.get('name', 'Unknown')
                if lec_name and lec_name != 'Unknown':
                    G.add_node(f"LEC_{lec_name}", type="lecturer", name=lec_name, title=lec.get('title', ''))
            
            # Thêm edges
            for lec in kb.lecturers:
                lec_name = lec.get('name', '')
                lec_text = lec.get('title', '') + " " + lec.get('research_area', '')
                for dept in kb.departments:
                    dept_name = dept.get('name', '')
                    if dept_name and dept_name.lower() in lec_text.lower():
                        node_name = f"DEPT_{dept_name}"
                        if G.has_node(node_name):
                            G.add_edge(f"LEC_{lec_name}", node_name, relation="belongs_to")
            
            self.graph = G
            logger.info(f"Knowledge Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.warning(f"Could not build knowledge graph: {e}")
            return None


# ========== MAIN ==========
def main():
    print("=" * 70)
    print("🚀 UTC KNOWLEDGE SYSTEM - Research Grade v2.1")
    print("=" * 70)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    print("\n[1] Starting advanced crawler...")
    crawler = AdvancedCrawler(BASE_URL, max_depth=CRAWL_DEPTH, max_pages=MAX_PAGES)
    kb = crawler.crawl()
    
    print("\n[2] Generating Q&A pairs...")
    kb.qa_pairs = QAGenerator.generate(kb)
    
    print("\n[3] Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder()
    graph = kg_builder.build(kb)
    
    print("\n[4] Saving knowledge base...")
    output_file = os.path.join(OUTPUT_DIR, "utc_knowledge_advanced.json")
    kb.save(output_file)
    
    print("\n" + "=" * 70)
    print("📊 CRAWL SUMMARY")
    print("=" * 70)
    print(f"📄 Pages crawled: {kb.metadata['total_pages']}")
    print(f"🏛️ Departments found: {kb.metadata['total_departments']}")
    print(f"👨‍🏫 Lecturers found: {kb.metadata['total_lecturers']}")
    print(f"🔬 Research projects: {kb.metadata['total_research']}")
    print(f"💬 Q&A pairs: {len(kb.qa_pairs)}")
    print(f"📰 News items: {len(kb.news)}")
    print("\n✅ COMPLETE!")
    print(f"📁 Data saved to: {output_file}")
    print(f"📝 Log saved to: {LOG_FILE}")
    
    return kb


if __name__ == "__main__":
    required_libs = ['requests', 'bs4', 'numpy']
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"❌ Missing: {missing}")
        print("Run: pip install requests beautifulsoup4 numpy")
        sys.exit(1)
    
    main()