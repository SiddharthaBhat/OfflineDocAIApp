package com.siddu.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val conversationId: Long,
    val role: String, // "USER" or "AI"
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val sources: String? = null, // JSON array of page numbers
    val thinkingDurationMs: Long? = null
)
