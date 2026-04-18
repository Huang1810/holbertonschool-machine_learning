#!/usr/bin/env python3
"""
Returns all students sorted by average score (descending)
Each student includes an added field: averageScore
"""

def top_students(mongo_collection):
    pipeline = [
        {
            "$project": {
                "name": 1,
                "averageScore": {
                    "$avg": "$topics.score"
                }
            }
        },
        {
            "$sort": {
                "averageScore": -1
            }
        }
    ]

    return list(mongo_collection.aggregate(pipeline))
