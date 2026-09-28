package com.siddu.offlinedocai.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ChunkDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(chunks: List<ChunkEntity>)

    @Query("SELECT * FROM chunks WHERE documentId = :docId")
    fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>>

    // Simple in‑memory similarity search will be implemented in the repository,
    // so we just expose a method to retrieve all chunks for a document.
    @Query("SELECT * FROM chunks WHERE documentId IN (:docIds)")
    suspend fun getAllChunksForDocumentIds(docIds: List<Long>): List<ChunkEntity>

    @Query("SELECT * FROM chunks WHERE documentId = :docId AND pageNumber = :pageNumber ORDER BY id ASC")
    suspend fun getChunksForDocumentAndPage(docId: Long, pageNumber: Int): List<ChunkEntity>

    @Query("SELECT * FROM chunks WHERE documentId = :docId AND pageNumber BETWEEN :startPage AND :endPage ORDER BY pageNumber ASC, id ASC")
    suspend fun getChunksForDocumentAndPageRange(docId: Long, startPage: Int, endPage: Int): List<ChunkEntity>
}
