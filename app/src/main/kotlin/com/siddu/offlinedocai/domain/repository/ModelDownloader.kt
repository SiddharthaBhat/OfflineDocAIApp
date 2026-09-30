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
            
            var currentUrl = urlStr
            var connection: HttpURLConnection? = null
            var redirects = 0
            val maxRedirects = 6

            while (redirects < maxRedirects) {
                val url = URL(currentUrl)
                connection = (url.openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true
                    connectTimeout = 30000
                    readTimeout = 30000
                    setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                }
                connection.connect()

                val code = connection.responseCode
                if (code == HttpURLConnection.HTTP_MOVED_PERM ||
                    code == HttpURLConnection.HTTP_MOVED_TEMP ||
                    code == HttpURLConnection.HTTP_SEE_OTHER ||
                    code == 307 || code == 308) {
                    val location = connection.getHeaderField("Location")
                    if (!location.isNullOrEmpty()) {
                        currentUrl = location
                        redirects++
                        connection.disconnect()
                        continue
                    }
                }
                break
            }

            val finalConn = connection ?: throw IllegalStateException("Failed to establish connection")
            if (finalConn.responseCode != HttpURLConnection.HTTP_OK) {
                _downloadState.value = DownloadState.Error("Server returned HTTP ${finalConn.responseCode}")
                return@withContext
            }

            val fileLength = finalConn.contentLengthLong
            finalConn.inputStream.use { input ->
                FileOutputStream(destinationFile).use { output ->
                    val data = ByteArray(65536) // 64KB buffer for high-throughput 1GB+ downloads
                    var total: Long = 0
                    var count: Int
                    var lastReportedPercent = -1
                    while (input.read(data).also { count = it } != -1) {
                        total += count
                        if (fileLength > 0) {
                            val percent = ((total * 100) / fileLength).toInt()
                            if (percent != lastReportedPercent) {
                                lastReportedPercent = percent
                                _downloadState.value = DownloadState.Progress(total, fileLength, percent)
                            }
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
