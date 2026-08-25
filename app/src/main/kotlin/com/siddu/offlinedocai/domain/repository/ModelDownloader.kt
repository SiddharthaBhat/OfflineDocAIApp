package com.siddu.offlinedocai.domain.repository

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

sealed class DownloadState {
    object Idle : DownloadState()
    data class Progress(val bytesDownloaded: Long, val totalBytes: Long, val percent: Int) : DownloadState()
    object Completed : DownloadState()
    data class Error(val message: String) : DownloadState()
}

class ModelDownloader(private val context: Context) {
    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState

    suspend fun downloadModel(urlStr: String, destinationFile: File) = withContext(Dispatchers.IO) {
        try {
            _downloadState.value = DownloadState.Progress(0, 100, 0)
            
            // Ensure parent directories exist
            destinationFile.parentFile?.mkdirs()
            
            val url = URL(urlStr)
            val connection = url.openConnection() as HttpURLConnection
            connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                _downloadState.value = DownloadState.Error("Server returned HTTP ${connection.responseCode}")
                return@withContext
            }

            val fileLength = connection.contentLengthLong
            connection.inputStream.use { input ->
                FileOutputStream(destinationFile).use { output ->
                    val data = ByteArray(8192)
                    var total: Long = 0
                    var count: Int
                    while (input.read(data).also { count = it } != -1) {
                        total += count
                        if (fileLength > 0) {
                            val percent = ((total * 100) / fileLength).toInt()
                            _downloadState.value = DownloadState.Progress(total, fileLength, percent)
                        }
                        output.write(data, 0, count)
                    }
                }
            }
            _downloadState.value = DownloadState.Completed
        } catch (e: Exception) {
            _downloadState.value = DownloadState.Error(e.message ?: "Unknown download error")
        }
    }
    
    fun resetState() {
        _downloadState.value = DownloadState.Idle
    }
}
