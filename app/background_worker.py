"""Background worker for processing tasks"""

import time
import asyncio
import logging
from typing import Dict, Any
from app.youtube_cover_detector import CoverDetector, logger
from multiprocessing import Process

logger = logging.getLogger(__name__)

def process_queue_forever(queue, active_tasks):
    """Process queue items forever in a separate process"""
    logger.info("Background worker started")
    
    # Create event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            if not queue.empty():
                logger.info(f"Queue processor waiting for tasks. Current queue size: {queue.qsize()}")
                request = queue.get()  # Get the next request from queue
                logger.info(f"Processing request {request['id']}")
                
                try:
                    detector = CoverDetector()
                    
                    # Update status to downloading
                    request['status'] = 'downloading_first'
                    request['progress'] = 20
                    request['message'] = 'Downloading first video...'
                    active_tasks[request['id']] = request
                    
                    # Run async method in the event loop
                    result = loop.run_until_complete(
                        detector.compare_videos(request['url1'], request['url2'], request)
                    )
                    
                    # Update status to completed
                    request['status'] = 'completed'
                    request['result'] = result
                    request['progress'] = 100
                    request['message'] = 'Analysis complete'
                    active_tasks[request['id']] = request
                    logger.info(f"Request {request['id']} completed with result: {result}")
                    
                except Exception as e:
                    request['status'] = 'failed'
                    request['error'] = str(e)
                    request['progress'] = 0
                    request['message'] = f'Error: {str(e)}'
                    active_tasks[request['id']] = request
                    logger.error(f"Error processing request {request['id']}: {e}")
            else:
                time.sleep(1)  # Sleep when queue is empty
                
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            time.sleep(1)

def start_background_worker(queue, active_tasks):
    """Start the background worker process"""
    # Create a new process to handle the queue
    process = Process(target=process_queue_forever, args=(queue, active_tasks))
    process.daemon = False  # Changed from True to False
    process.start()
    logger.info(f"Started background worker process with PID {process.pid}") 