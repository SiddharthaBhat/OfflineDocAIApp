package com.siddu.offlinedocai.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.siddu.offlinedocai.data.entities.ConversationEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ConversationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(conversation: ConversationEntity): Long

    @Query("SELECT * FROM conversations ORDER BY createdAt DESC")
    fun getAllConversations(): Flow<List<ConversationEntity>>

    @Query("SELECT * FROM conversations WHERE id = :id LIMIT 1")
    suspend fun getById(id: Long): ConversationEntity?

    @androidx.room.Delete
    suspend fun delete(conversation: ConversationEntity)
}
