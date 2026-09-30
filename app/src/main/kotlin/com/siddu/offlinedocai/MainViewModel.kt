package com.siddu.offlinedocai

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.ModelDownloader
import com.siddu.offlinedocai.domain.repository.DownloadState
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import com.siddu.offlinedocai.ai.queue.QuestionQueueManager
import com.siddu.offlinedocai.ai.queue.QuestionQueueState
import com.siddu.offlinedocai.ai.queue.QuestionQueueParser
import java.io.File

/**
 * Central ViewModel for the OfflineDocAI app.
 * Provides access to the Room database, model downloads (Embedding & LLM), and exposes StateFlows for UI observation.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val database: AppDatabase = Room.databaseBuilder(
        application,
        AppDatabase::class.java,
        "offline_docai.db"
    ).addMigrations(AppDatabase.MIGRATION_2_3)
     .fallbackToDestructiveMigration()
     .build()

    private val documentDao = database.documentDao()
    private val conversationDao = database.conversationDao()
    private val messageDao = database.messageDao()

    private val _documents = MutableStateFlow<List<DocumentEntity>>(emptyList())
    val documents: StateFlow<List<DocumentEntity>> = _documents

    private val _conversations = MutableStateFlow<List<ConversationEntity>>(emptyList())
    val conversations: StateFlow<List<ConversationEntity>> = _conversations

    val isEmbeddingModelDownloaded = MutableStateFlow(false)
    val isLlmModelDownloaded = MutableStateFlow(false)

    // Embedding Model Management
    private val embeddingDownloader = ModelDownloader(application)
    val embeddingDownloadState: StateFlow<DownloadState> = embeddingDownloader.downloadState
    val currentlyDownloadingEmbeddingModel = MutableStateFlow<String?>(null)
    val downloadedEmbeddingModels = MutableStateFlow<List<File>>(emptyList())
    val selectedEmbeddingModel = MutableStateFlow<File?>(null)

    // LLM Model Management
    private val llmDownloader = ModelDownloader(application)
    val llmDownloadState: StateFlow<DownloadState> = llmDownloader.downloadState
    val currentlyDownloadingModel = MutableStateFlow<String?>(null)
    val downloadedLlmModels = MutableStateFlow<List<File>>(emptyList())
    val selectedLlmModel = MutableStateFlow<File?>(null)

    // Repositories
    private val documentRepository = DocumentRepository(application, database)
    private val vectorStore = VectorStore(database.chunkDao())
    private val chatRepository = ChatRepository(application, database, vectorStore)

    val importState: StateFlow<DocumentRepository.ImportState> = documentRepository.importState
    val chatState: StateFlow<ChatRepository.ChatState> = chatRepository.chatState

    // Question Queue Manager for batch/sequential question execution
    val questionQueueManager = QuestionQueueManager(viewModelScope)
    val questionQueueState: StateFlow<QuestionQueueState> = questionQueueManager.queueState

    init {
        // Load existing data from DB
        viewModelScope.launch {
            documentDao.getAllDocuments().collect { docs ->
                _documents.value = docs
                // Automatically recover from previous session crashes: mark orphaned PROCESSING docs as FAILED
                docs.filter { it.status == "PROCESSING" }.forEach { stuck ->
                    documentDao.update(stuck.copy(status = "FAILED"))
                }
            }
        }
        viewModelScope.launch {
            conversationDao.getAllConversations().collect { _conversations.value = it }
        }
        checkModels()
        autoDownloadEmbeddingIfMissing()
    }

    fun checkModels() {
        val embedDir = File(getApplication<Application>().filesDir, "models/embeddings")
        if (!embedDir.exists()) embedDir.mkdirs()

        val embedFiles = embedDir.listFiles()?.filter {
            it.extension.equals("onnx", ignoreCase = true) ||
            it.extension.equals("gguf", ignoreCase = true)
        } ?: emptyList()
        downloadedEmbeddingModels.value = embedFiles
        isEmbeddingModelDownloaded.value = embedFiles.isNotEmpty()

        val prefs = getApplication<Application>().getSharedPreferences("docai_models", Context.MODE_PRIVATE)
        val savedEmbedName = prefs.getString("selected_embedding_model", null)
        val preferredEmbed = embedFiles.firstOrNull { it.name == savedEmbedName }
            ?: embedFiles.firstOrNull { it.extension.equals("onnx", ignoreCase = true) }
            ?: embedFiles.firstOrNull()
        selectedEmbeddingModel.value = preferredEmbed

        val llmDir = File(getApplication<Application>().filesDir, "models/llm")
        if (!llmDir.exists()) llmDir.mkdirs()

        val llmFiles = llmDir.listFiles()?.filter { it.extension.equals("gguf", ignoreCase = true) } ?: emptyList()
        downloadedLlmModels.value = llmFiles
        isLlmModelDownloaded.value = llmFiles.isNotEmpty()

        val savedLlmName = prefs.getString("selected_llm_model", null)
        val preferredLlm = llmFiles.firstOrNull { it.name == savedLlmName } ?: llmFiles.firstOrNull()
        selectedLlmModel.value = preferredLlm
    }

    fun selectEmbeddingModel(file: File) {
        selectedEmbeddingModel.value = file
        val prefs = getApplication<Application>().getSharedPreferences("docai_models", Context.MODE_PRIVATE)
        prefs.edit().putString("selected_embedding_model", file.name).apply()
    }

    fun deleteEmbeddingModel(file: File) {
        viewModelScope.launch {
            if (file.exists()) {
                file.delete()
            }
            val prefs = getApplication<Application>().getSharedPreferences("docai_models", Context.MODE_PRIVATE)
            if (prefs.getString("selected_embedding_model", null) == file.name) {
                prefs.edit().remove("selected_embedding_model").apply()
            }
            checkModels()
        }
    }

    fun downloadEmbeddingModel(urlStr: String, fileName: String) {
        currentlyDownloadingEmbeddingModel.value = fileName
        val destFile = File(getApplication<Application>().filesDir, "models/embeddings/$fileName")
        viewModelScope.launch {
            embeddingDownloader.downloadModel(urlStr, destFile)
            checkModels()
            if (selectedEmbeddingModel.value == null && destFile.exists()) {
                selectEmbeddingModel(destFile)
            }
        }
    }

    fun selectLlmModel(file: File) {
        selectedLlmModel.value = file
        val prefs = getApplication<Application>().getSharedPreferences("docai_models", Context.MODE_PRIVATE)
        prefs.edit().putString("selected_llm_model", file.name).apply()
    }

    fun deleteLlmModel(file: File) {
        viewModelScope.launch {
            if (file.exists()) {
                file.delete()
            }
            val prefs = getApplication<Application>().getSharedPreferences("docai_models", Context.MODE_PRIVATE)
            if (prefs.getString("selected_llm_model", null) == file.name) {
                prefs.edit().remove("selected_llm_model").apply()
            }
            checkModels()
        }
    }

    private fun autoDownloadEmbeddingIfMissing() {
        if (!isEmbeddingModelDownloaded.value) {
            val defaultModel = AvailableEmbeddingModels.list.firstOrNull()
            if (defaultModel != null) {
                downloadEmbeddingModel(defaultModel.downloadUrl, defaultModel.fileName)
            }
        }
    }

    fun downloadLlmModel(urlStr: String, fileName: String) {
        currentlyDownloadingModel.value = fileName
        val destFile = File(getApplication<Application>().filesDir, "models/llm/$fileName")
        viewModelScope.launch {
            llmDownloader.downloadModel(urlStr, destFile)
            checkModels()
            if (selectedLlmModel.value == null && destFile.exists()) {
                selectLlmModel(destFile)
            }
        }
    }

    fun resetLlmDownloadState() {
        currentlyDownloadingModel.value = null
        llmDownloader.resetState()
    }

    fun resetEmbeddingDownloadState() {
        currentlyDownloadingEmbeddingModel.value = null
        embeddingDownloader.resetState()
    }

    /** Trigger document import */
    fun importDocument(uri: Uri) {
        documentRepository.importDocument(uri)
    }

    /** Start a new conversation linked to selected document IDs */
    suspend fun startConversation(title: String = "New Chat", documentIds: List<Long> = emptyList()): Long {
        return chatRepository.startConversation(title, documentIds.firstOrNull())
    }

    /** Send a message in a conversation */
    fun sendMessage(convId: Long, text: String, docIds: List<Long>) {
        chatRepository.sendMessage(convId, text, docIds)
    }

    /** Start sequential processing of multiple questions */
    fun startQuestionQueue(convId: Long, questions: List<String>, docIds: List<Long>) {
        questionQueueManager.startQueue(convId, questions, docIds) { targetConvId, questionText, targetDocIds ->
            chatRepository.executeMessage(targetConvId, questionText, targetDocIds)
        }
    }

    fun pauseQuestionQueue() {
        questionQueueManager.pauseQueue()
    }

    fun resumeQuestionQueue() {
        questionQueueManager.resumeQueue()
    }

    fun stopQuestionQueue() {
        questionQueueManager.stopQueue()
    }

    fun clearQuestionQueue() {
        questionQueueManager.clearQueue()
    }

    fun parseQuestions(rawInput: String): List<String> {
        return QuestionQueueParser.parseQuestions(rawInput)
    }

    // Delete a document
    fun deleteDocument(documentId: Long) {
        viewModelScope.launch {
            documentDao.getById(documentId)?.let {
                documentDao.delete(it)
            }
        }
    }

    // Delete a conversation and its messages
    fun deleteConversation(convId: Long) {
        viewModelScope.launch {
            conversationDao.getById(convId)?.let {
                conversationDao.delete(it)
            }
            messageDao.deleteMessagesForConversation(convId)
        }
    }

    // Get messages for a conversation as StateFlow
    fun messagesForConversation(convId: Long): StateFlow<List<MessageEntity>> {
        return messageDao.getMessagesForConversation(convId).stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    }
}
