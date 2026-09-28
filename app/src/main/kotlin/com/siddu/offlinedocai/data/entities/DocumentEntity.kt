package com.siddu.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val uri: String,
    val name: String,
    val type: String,
    val pageCount: Int = 0,
    val addedAt: Long = Date().time,
    val status: String = "PROCESSING",
    val profileJson: String? = null
)
