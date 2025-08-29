"""
Database module for people counting system.
Handles SQLite operations with WAL mode for concurrent access.
"""
import sqlite3
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
import aiosqlite

logger = logging.getLogger(__name__)

class CounterDB:
    """SQLite database handler for people counting events."""
    
    def __init__(self, db_path: str = "counts.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def init_db(self) -> None:
        """Initialize database schema and enable WAL mode."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrent access
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            await db.execute("PRAGMA temp_store=memory")
            
            # Create crossings table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS crossings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    track_id INTEGER NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
                    confidence REAL NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    frame_number INTEGER,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossings_timestamp 
                ON crossings(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossings_track_direction 
                ON crossings(track_id, direction)
            """)
            
            await db.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    async def log_crossing(self, track_id: int, direction: str, 
                          confidence: float, x: float, y: float,
                          frame_number: Optional[int] = None) -> None:
        """Log a crossing event to database.
        
        Args:
            track_id: Unique track identifier
            direction: 'in' or 'out'
            confidence: Detection confidence score
            x, y: Crossing point coordinates
            frame_number: Optional frame number
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO crossings 
                    (track_id, direction, confidence, x, y, frame_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (track_id, direction, confidence, x, y, frame_number))
                await db.commit()
                
            logger.debug(f"Logged crossing: track={track_id}, dir={direction}, "
                        f"conf={confidence:.2f}, pos=({x:.1f},{y:.1f})")
                        
        except Exception as e:
            logger.error(f"Failed to log crossing: {e}")
    
    async def get_today_counts(self) -> Dict[str, int]:
        """Get today's crossing counts.
        
        Returns:
            Dictionary with 'in', 'out', and 'occupancy' counts
        """
        today = date.today().isoformat()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get counts for today
                cursor = await db.execute("""
                    SELECT direction, COUNT(*) 
                    FROM crossings 
                    WHERE DATE(timestamp) = ?
                    GROUP BY direction
                """, (today,))
                
                rows = await cursor.fetchall()
                
                counts = {'in': 0, 'out': 0}
                for direction, count in rows:
                    counts[direction] = count
                
                counts['occupancy'] = counts['in'] - counts['out']
                return counts
                
        except Exception as e:
            logger.error(f"Failed to get today's counts: {e}")
            return {'in': 0, 'out': 0, 'occupancy': 0}
    
    async def get_counts_by_date(self, target_date: str) -> Dict[str, int]:
        """Get crossing counts for a specific date.
        
        Args:
            target_date: Date string in YYYY-MM-DD format
            
        Returns:
            Dictionary with 'in', 'out', and 'occupancy' counts
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT direction, COUNT(*) 
                    FROM crossings 
                    WHERE DATE(timestamp) = ?
                    GROUP BY direction
                """, (target_date,))
                
                rows = await cursor.fetchall()
                
                counts = {'in': 0, 'out': 0}
                for direction, count in rows:
                    counts[direction] = count
                
                counts['occupancy'] = counts['in'] - counts['out']
                return counts
                
        except Exception as e:
            logger.error(f"Failed to get counts for {target_date}: {e}")
            return {'in': 0, 'out': 0, 'occupancy': 0}
    
    async def get_recent_crossings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent crossing events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of crossing event dictionaries
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT timestamp, track_id, direction, confidence, x, y, frame_number
                    FROM crossings 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                
                crossings = []
                for row in rows:
                    crossings.append({
                        'timestamp': row[0],
                        'track_id': row[1],
                        'direction': row[2],
                        'confidence': row[3],
                        'x': row[4],
                        'y': row[5],
                        'frame_number': row[6]
                    })
                
                return crossings
                
        except Exception as e:
            logger.error(f"Failed to get recent crossings: {e}")
            return []
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Remove crossing data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    DELETE FROM crossings 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                deleted_count = cursor.rowcount
                await db.commit()
                
                logger.info(f"Cleaned up {deleted_count} old crossing records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

# Synchronous wrapper for compatibility
class SyncCounterDB:
    """Synchronous wrapper for CounterDB."""
    
    def __init__(self, db_path: str = "counts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def init_db(self) -> None:
        """Initialize database schema and enable WAL mode."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            # Create schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crossings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    track_id INTEGER NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
                    confidence REAL NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    frame_number INTEGER,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossings_timestamp 
                ON crossings(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossings_track_direction 
                ON crossings(track_id, direction)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def log_crossing(self, track_id: int, direction: str, 
                    confidence: float, x: float, y: float,
                    frame_number: Optional[int] = None) -> None:
        """Log a crossing event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO crossings 
                    (track_id, direction, confidence, x, y, frame_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (track_id, direction, confidence, x, y, frame_number))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log crossing: {e}")
    
    def get_today_counts(self) -> Dict[str, int]:
        """Get today's crossing counts."""
        today = date.today().isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT direction, COUNT(*) 
                    FROM crossings 
                    WHERE DATE(timestamp) = ?
                    GROUP BY direction
                """, (today,))
                
                rows = cursor.fetchall()
                
                counts = {'in': 0, 'out': 0}
                for direction, count in rows:
                    counts[direction] = count
                
                counts['occupancy'] = counts['in'] - counts['out']
                return counts
                
        except Exception as e:
            logger.error(f"Failed to get today's counts: {e}")
            return {'in': 0, 'out': 0, 'occupancy': 0}
