package com.siddu.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * BUG 8 FIX: Override equals()/hashCode()/toString() for correct ByteArray structural comparison.
 *
 * Kotlin data classes generate equals()/hashCode() that delegate to Array.equals() for arrays,
 * which for ByteArray uses reference equality (not content equality). This breaks Room's
 * conflict detection on REPLACE and can cause stale embedding bytes to be silently reused.
 *
 * The fix uses contentEquals() for ByteArray comparison and Arrays.hashCode() for hashing.
 */
@Entity(tableName = "chunks")
data class ChunkEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val documentId: Long,
    val pageNumber: Int,
    val text: String,
    var embedding: ByteArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ChunkEntity) return false
        return id == other.id &&
            documentId == other.documentId &&
            pageNumber == other.pageNumber &&
            text == other.text &&
            (embedding contentEquals other.embedding)
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + documentId.hashCode()
        result = 31 * result + pageNumber
        result = 31 * result + text.hashCode()
        result = 31 * result + (embedding?.contentHashCode() ?: 0)
        return result
    }

    override fun toString(): String =
        "ChunkEntity(id=$id, documentId=$documentId, pageNumber=$pageNumber, " +
        "text='${text.take(40)}...', embedding=${if (embedding != null) "[${embedding!!.size} bytes]" else "null"})"
}
