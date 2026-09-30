package com.siddu.offlinedocai

import android.os.Bundle
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.FormatAlignLeft
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.PlaylistAdd
import androidx.compose.material.icons.filled.PlaylistPlay
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.ui.platform.LocalContext
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.*
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.DownloadState
import com.siddu.offlinedocai.ai.llm.AvailableLlmModel
import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            com.tom_roush.pdfbox.android.PDFBoxResourceLoader.init(applicationContext)
        } catch (_: Exception) {}
        setContent { OfflineDocAIApp() }
    }

    override fun dispatchGenericMotionEvent(ev: android.view.MotionEvent?): Boolean {
        return try {
            super.dispatchGenericMotionEvent(ev)
        } catch (e: Exception) {
            // Catches AndroidComposeView hover event bug (Google Issue #314269723)
            android.util.Log.w("MainActivity", "Intercepted generic motion event exception: ${e.message}")
            true
        }
    }

    override fun dispatchTouchEvent(ev: android.view.MotionEvent?): Boolean {
        return try {
            super.dispatchTouchEvent(ev)
        } catch (e: Exception) {
            android.util.Log.w("MainActivity", "Intercepted touch event exception: ${e.message}")
            true
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OfflineDocAIApp(mainViewModel: MainViewModel = viewModel()) {
    val navController = rememberNavController()
    val items = listOf(
        NavItem("home", "Home", Icons.Default.Home),
        NavItem("chats", "Chats", Icons.Default.Chat),
        NavItem("documents", "Documents", Icons.Default.Description),
        NavItem("settings", "Settings", Icons.Default.Settings)
    )
    Scaffold(
        bottomBar = {
            NavigationBar {
                val currentDestination = navController.currentBackStackEntryAsState().value?.destination
                items.forEach { item ->
                    NavigationBarItem(
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label) },
                        selected = currentDestination?.hierarchy?.any { it.route == item.route } == true,
                        onClick = {
                            navController.navigate(item.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "home",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("home") { HomeScreen(mainViewModel, navController) }
            composable("chats") { ChatsScreen(mainViewModel, navController) }
            composable("documents") { DocumentsScreen(mainViewModel, navController) }
            composable("settings") { SettingsScreen(mainViewModel) }
            composable("documentDetail/{docId}") { backStackEntry ->
                val docId = backStackEntry.arguments?.getString("docId")?.toLongOrNull() ?: return@composable
                DocumentDetailScreen(docId, mainViewModel, navController)
            }
            composable("chat/{convId}") { backStackEntry ->
                val convId = backStackEntry.arguments?.getString("convId")?.toLongOrNull() ?: return@composable
                ChatScreen(convId, mainViewModel, navController)
            }
            composable("newChat") { NewChatScreen(mainViewModel, navController) }
            composable("importProcessing") { ImportProcessingScreen(mainViewModel, navController) }
        }
    }
}

data class NavItem(val route: String, val label: String, val icon: ImageVector)

@Composable
fun HomeScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val conversations by viewModel.conversations.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val coroutineScope = rememberCoroutineScope()

    // Trigger check on layout load
    LaunchedEffect(Unit) {
        viewModel.checkModels()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        Text("Offline DocAI", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(4.dp))
        Text("Your secure, fully local RAG assistant", style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
        Spacer(modifier = Modifier.height(24.dp))

        // Check if models need to be downloaded
        if (!isEmbedDownloaded || !isLlmDownloaded) {
            ModelDownloadBanner(viewModel)
            Spacer(modifier = Modifier.height(16.dp))
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("documents") }
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("${documents.size}", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Documents", style = MaterialTheme.typography.bodyMedium)
                }
            }
            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("chats") }
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("${conversations.size}", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Chats", style = MaterialTheme.typography.bodyMedium)
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                coroutineScope.launch {
                    val convId = viewModel.startConversation(
                        title = "General AI Chat",
                        documentIds = emptyList()
                    )
                    navController.navigate("chat/$convId")
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = isLlmDownloaded
        ) {
            Icon(Icons.Default.Chat, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("🤖 Start General AI Chat (No Document)")
        }

        Spacer(modifier = Modifier.height(10.dp))

        OutlinedButton(
            onClick = { navController.navigate("newChat") },
            modifier = Modifier.fillMaxWidth(),
            enabled = isLlmDownloaded
        ) {
            Icon(Icons.Default.Description, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("📄 Chat with a Document (RAG)")
        }
    }
}

@Composable
fun ModelOptionCard(
    model: AvailableLlmModel,
    isDownloaded: Boolean,
    isSelected: Boolean,
    isDownloading: Boolean,
    downloadProgress: DownloadState.Progress?,
    onDownload: () -> Unit,
    onSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isSelected -> MaterialTheme.colorScheme.primaryContainer
                isDownloaded -> MaterialTheme.colorScheme.surfaceVariant
                else -> MaterialTheme.colorScheme.surface
            }
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isSelected) androidx.compose.foundation.BorderStroke(2.dp, MaterialTheme.colorScheme.primary) else null,
        modifier = modifier
            .fillMaxWidth()
            .then(
                if (isDownloaded) Modifier.clickable { onSelect() } else Modifier
            )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${model.parameterSize} params • Size: ${model.downloadSize}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = when {
                        isSelected -> MaterialTheme.colorScheme.primary
                        isDownloaded -> MaterialTheme.colorScheme.secondaryContainer
                        else -> MaterialTheme.colorScheme.tertiaryContainer
                    }
                ) {
                    Text(
                        text = when {
                            isSelected -> "Active"
                            isDownloaded -> "Ready"
                            else -> model.badge
                        },
                        color = when {
                            isSelected -> MaterialTheme.colorScheme.onPrimary
                            isDownloaded -> MaterialTheme.colorScheme.onSecondaryContainer
                            else -> MaterialTheme.colorScheme.onTertiaryContainer
                        },
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = model.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "⚡ Recommended: ${model.ramRequirement}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )

            if (isDownloading && downloadProgress != null) {
                Spacer(modifier = Modifier.height(10.dp))
                LinearProgressIndicator(
                    progress = { downloadProgress.percent / 100f },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Downloading: ${downloadProgress.percent}% (${downloadProgress.bytesDownloaded / 1024 / 1024} MB / ${downloadProgress.totalBytes / 1024 / 1024} MB)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            } else if (!isDownloaded) {
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = onDownload,
                    enabled = !isDownloading,
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(vertical = 6.dp)
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Download ${model.downloadSize}")
                }
            } else if (!isSelected) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Tap to switch to this model for chat",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
fun EmbeddingModelOptionCard(
    model: AvailableEmbeddingModel,
    isDownloaded: Boolean,
    isSelected: Boolean,
    isDownloading: Boolean,
    downloadProgress: DownloadState.Progress?,
    onDownload: () -> Unit,
    onSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isSelected -> MaterialTheme.colorScheme.primaryContainer
                isDownloaded -> MaterialTheme.colorScheme.surfaceVariant
                else -> MaterialTheme.colorScheme.surface
            }
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isSelected) androidx.compose.foundation.BorderStroke(2.dp, MaterialTheme.colorScheme.primary) else null,
        modifier = modifier
            .fillMaxWidth()
            .then(
                if (isDownloaded) Modifier.clickable { onSelect() } else Modifier
            )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${model.format} • ${model.dimension}-dim • Size: ${model.downloadSize}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = when {
                        isSelected -> MaterialTheme.colorScheme.primary
                        isDownloaded -> MaterialTheme.colorScheme.secondaryContainer
                        else -> MaterialTheme.colorScheme.tertiaryContainer
                    }
                ) {
                    Text(
                        text = when {
                            isSelected -> "Active"
                            isDownloaded -> "Ready"
                            else -> model.badge
                        },
                        color = when {
                            isSelected -> MaterialTheme.colorScheme.onPrimary
                            isDownloaded -> MaterialTheme.colorScheme.onSecondaryContainer
                            else -> MaterialTheme.colorScheme.onTertiaryContainer
                        },
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = model.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (isDownloading && downloadProgress != null) {
                Spacer(modifier = Modifier.height(10.dp))
                LinearProgressIndicator(
                    progress = { downloadProgress.percent / 100f },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Downloading: ${downloadProgress.percent}% (${downloadProgress.bytesDownloaded / 1024 / 1024} MB / ${downloadProgress.totalBytes / 1024 / 1024} MB)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            } else if (!isDownloaded) {
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = onDownload,
                    enabled = !isDownloading,
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(vertical = 6.dp)
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Download ${model.downloadSize}")
                }
            } else if (!isSelected) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Tap to switch to this model for embeddings",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
fun ModelDownloadBanner(viewModel: MainViewModel) {
    val embedDownloadState by viewModel.embeddingDownloadState.collectAsState()
    val llmDownloadState by viewModel.llmDownloadState.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val currentlyDownloading by viewModel.currentlyDownloadingModel.collectAsState()
    val downloadedLlmModels by viewModel.downloadedLlmModels.collectAsState()
    val selectedLlmModel by viewModel.selectedLlmModel.collectAsState()

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.8f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Action Required: Models Missing", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            if (!isEmbedDownloaded) {
                Text("• Embedding Model (all-MiniLM-L6-v2 ONNX):", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(4.dp))
                when (embedDownloadState) {
                    is DownloadState.Idle -> {
                        Text("Waiting to start...")
                    }
                    is DownloadState.Progress -> {
                        val progress = embedDownloadState as DownloadState.Progress
                        Column {
                            LinearProgressIndicator(
                                progress = { progress.percent / 100f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Downloading: ${progress.percent}% (${(progress.bytesDownloaded / 1024 / 1024)}MB / ${(progress.totalBytes / 1024 / 1024)}MB)", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    is DownloadState.Completed -> {
                        Text("Finished!", color = Color(0xFF4CAF50))
                    }
                    is DownloadState.Error -> {
                        val error = embedDownloadState as DownloadState.Error
                        Text("Error: ${error.message}", color = Color.Red, style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(4.dp))
                        val defaultEmbed = AvailableEmbeddingModels.list.firstOrNull()
                        if (defaultEmbed != null) {
                            Button(onClick = { viewModel.downloadEmbeddingModel(defaultEmbed.downloadUrl, defaultEmbed.fileName) }) {
                                Text("Retry Embedding Download")
                            }
                        }
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
            }

            if (!isLlmDownloaded) {
                Text("• Local LLM (SLM): Required for chatting", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(4.dp))

                when (llmDownloadState) {
                    is DownloadState.Idle -> {
                        Text("Choose an offline model to download:", style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(8.dp))
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            AvailableLlmModels.list.forEach { model ->
                                val isDownloaded = downloadedLlmModels.any { it.name == model.fileName }
                                val isSelected = selectedLlmModel?.name == model.fileName
                                ModelOptionCard(
                                    model = model,
                                    isDownloaded = isDownloaded,
                                    isSelected = isSelected,
                                    isDownloading = false,
                                    downloadProgress = null,
                                    onDownload = {
                                        viewModel.downloadLlmModel(model.downloadUrl, model.fileName)
                                    },
                                    onSelect = {
                                        val file = downloadedLlmModels.firstOrNull { it.name == model.fileName }
                                        if (file != null) viewModel.selectLlmModel(file)
                                    }
                                )
                            }
                        }
                    }
                    is DownloadState.Progress -> {
                        val progress = llmDownloadState as DownloadState.Progress
                        val modelInfo = AvailableLlmModels.list.firstOrNull { it.fileName == currentlyDownloading }
                        val displayName = modelInfo?.name ?: currentlyDownloading ?: "Local Model"
                        Column {
                            Text("Downloading $displayName...", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodyMedium)
                            Spacer(modifier = Modifier.height(4.dp))
                            LinearProgressIndicator(
                                progress = { progress.percent / 100f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Downloading: ${progress.percent}% (${(progress.bytesDownloaded / 1024 / 1024)}MB / ${(progress.totalBytes / 1024 / 1024)}MB)", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    is DownloadState.Completed -> {
                        Text("Download finished! Initializing...", style = MaterialTheme.typography.bodyMedium, color = Color(0xFF4CAF50))
                        Button(onClick = { viewModel.resetLlmDownloadState() }) {
                            Text("OK")
                        }
                    }
                    is DownloadState.Error -> {
                        val error = llmDownloadState as DownloadState.Error
                        Text("Error: ${error.message}", color = Color.Red, style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = { viewModel.resetLlmDownloadState() }) {
                            Text("Retry")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DocumentsScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.importDocument(it)
            navController.navigate("importProcessing")
        }
    }

    Scaffold(
        floatingActionButton = {
            if (isEmbedDownloaded) {
                FloatingActionButton(onClick = { filePicker.launch("*/*") }) {
                    Icon(Icons.Default.Add, contentDescription = "Import Document")
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            Text("Documents", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(16.dp))

            if (!isEmbedDownloaded) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Text("Please wait for the embedding model to finish downloading before importing documents.", modifier = Modifier.padding(16.dp))
                }
            } else if (documents.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No documents imported yet.", color = Color.Gray)
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(documents) { doc ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { navController.navigate("documentDetail/${doc.id}") }
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(doc.name, fontWeight = FontWeight.SemiBold)
                                    Spacer(modifier = Modifier.height(4.dp))
                                    val isReady = doc.status == "COMPLETED"
                                    val isFailed = doc.status == "FAILED"
                                    val statusColor = if (isReady) Color(0xFF4CAF50) else if (isFailed) Color(0xFFE53935) else Color(0xFFFF9800)
                                    val statusText = when (doc.status) {
                                        "COMPLETED" -> "Ready for RAG"
                                        "FAILED" -> "Failed / Incomplete • Delete & re-import"
                                        else -> "Processing..."
                                    }
                                    Text(
                                        text = "${doc.pageCount} pages • $statusText",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = statusColor
                                    )
                                }
                                IconButton(onClick = { viewModel.deleteDocument(doc.id) }) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete", tint = Color.Red)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentDetailScreen(docId: Long, viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val doc = documents.find { it.id == docId }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(doc?.name ?: "Document Detail") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            if (doc == null) {
                Text("Document not found.")
            } else {
                Text("File Name: ${doc.name}", style = MaterialTheme.typography.titleLarge)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Status: ${doc.status}", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Pages: ${doc.pageCount}", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("URI: ${doc.uri}", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        }
    }
}

@Composable
fun ImportProcessingScreen(viewModel: MainViewModel, navController: NavHostController) {
    val importState by viewModel.importState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Importing Document", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(32.dp))

        when (importState) {
            is DocumentRepository.ImportState.Idle -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Initializing...")
            }
            is DocumentRepository.ImportState.Reading -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Reading text content...")
            }
            is DocumentRepository.ImportState.Chunking -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Splitting document pages into semantic chunks...")
            }
            is DocumentRepository.ImportState.Embedding -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Generating vector embeddings locally...")
            }
            is DocumentRepository.ImportState.Saving -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Saving vectors to local SQLite store...")
            }
            is DocumentRepository.ImportState.Completed -> {
                Text("🎉 Import Completed Successfully!", style = MaterialTheme.typography.titleMedium, color = Color(0xFF4CAF50))
                Spacer(modifier = Modifier.height(32.dp))
                Button(onClick = { navController.navigate("documents") { popUpTo("home") } }) {
                    Text("Go to Documents")
                }
            }
            is DocumentRepository.ImportState.Error -> {
                val errMsg = (importState as DocumentRepository.ImportState.Error).message
                Text("❌ Import Failed", style = MaterialTheme.typography.titleMedium, color = Color.Red)
                Spacer(modifier = Modifier.height(8.dp))
                Text(errMsg, color = Color.Gray)
                Spacer(modifier = Modifier.height(32.dp))
                Button(onClick = { navController.popBackStack() }) {
                    Text("Go Back")
                }
            }
        }
    }
}

@Composable
fun ChatsScreen(viewModel: MainViewModel, navController: NavHostController) {
    val conversations by viewModel.conversations.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        floatingActionButton = {
            if (isLlmDownloaded) {
                FloatingActionButton(onClick = { navController.navigate("newChat") }) {
                    Icon(Icons.Default.Add, contentDescription = "New Chat")
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            Text("Conversations", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(16.dp))

            if (!isLlmDownloaded) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Text("Please download a language model on the Home screen to start chatting.", modifier = Modifier.padding(16.dp))
                }
            } else if (conversations.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("No chats active yet.", color = Color.Gray)
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(onClick = {
                            coroutineScope.launch {
                                val convId = viewModel.startConversation("General AI Chat", emptyList())
                                navController.navigate("chat/$convId")
                            }
                        }) {
                            Icon(Icons.Default.Chat, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Start a General AI Chat")
                        }
                    }
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(conversations) { chat ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { navController.navigate("chat/${chat.id}") }
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = if (chat.documentId == null) Icons.Default.Chat else Icons.Default.Description,
                                        contentDescription = null,
                                        tint = if (chat.documentId == null) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary
                                    )
                                    Spacer(modifier = Modifier.width(16.dp))
                                    Column {
                                        Text(chat.title, fontWeight = FontWeight.SemiBold)
                                        Spacer(modifier = Modifier.height(2.dp))
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Surface(
                                                color = if (chat.documentId == null) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.primaryContainer,
                                                shape = RoundedCornerShape(4.dp)
                                            ) {
                                                Text(
                                                    text = if (chat.documentId == null) "🤖 Pure AI" else "📄 RAG Doc",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                                    color = if (chat.documentId == null) MaterialTheme.colorScheme.onSecondaryContainer else MaterialTheme.colorScheme.onPrimaryContainer
                                                )
                                            }
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text(
                                                "Created at: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm").format(java.util.Date(chat.createdAt))}",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = Color.Gray
                                            )
                                        }
                                    }
                                }
                                IconButton(onClick = { viewModel.deleteConversation(chat.id) }) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete Conversation", tint = Color.Red)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewChatScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    var selectedTab by remember { mutableIntStateOf(0) } // 0 = General AI, 1 = Document RAG
    var title by remember { mutableStateOf("") }
    val selectedDocIds = remember { mutableStateListOf<Long>() }
    val coroutineScope = rememberCoroutineScope()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val selectedModel by viewModel.selectedLlmModel.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("New Conversation") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            TabRow(selectedTabIndex = selectedTab) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("🤖 General AI Chat") }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("📄 Document RAG") }
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            if (selectedTab == 0) {
                // General AI Chat Mode (No document)
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.4f)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "Direct AI Model Chat",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            "Chat directly with the installed model (${selectedModel?.name ?: "Qwen 2.5"}). No document is referenced. You can ask general questions, request explanations, coding help, or brainstorm freely!",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                Spacer(modifier = Modifier.height(20.dp))

                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    label = { Text("Chat Title (Optional)") },
                    placeholder = { Text("e.g. General AI Chat") },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.weight(1f))

                Button(
                    onClick = {
                        val finalTitle = title.ifBlank { "General AI Chat" }
                        coroutineScope.launch {
                            val convId = viewModel.startConversation(finalTitle, emptyList())
                            navController.navigate("chat/$convId") {
                                popUpTo("chats") { inclusive = false }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = isLlmDownloaded
                ) {
                    Icon(Icons.Default.Chat, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start General AI Chat")
                }
            } else {
                // Document RAG Chat Mode
                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    label = { Text("Chat Title (Optional)") },
                    placeholder = { Text("e.g. Document Discussion") },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(16.dp))
                Text("Select Document to Reference:", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(modifier = Modifier.height(8.dp))

                val completedDocs = documents.filter { it.status == "COMPLETED" }
                if (completedDocs.isEmpty()) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                    ) {
                        Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("No indexed documents found. Import documents in the 'Documents' tab first.", color = Color.Gray)
                            Spacer(modifier = Modifier.height(12.dp))
                            Button(onClick = { navController.navigate("documents") }) {
                                Text("Go to Documents")
                            }
                        }
                    }
                    Spacer(modifier = Modifier.weight(1f))
                } else {
                    LazyColumn(modifier = Modifier.weight(1f)) {
                        items(completedDocs) { doc ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        if (selectedDocIds.contains(doc.id)) {
                                            selectedDocIds.remove(doc.id)
                                        } else {
                                            selectedDocIds.clear()
                                            selectedDocIds.add(doc.id)
                                            if (title.isBlank()) {
                                                title = "${doc.name.substringBeforeLast(".")} Chat"
                                            }
                                        }
                                    }
                                    .padding(vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Checkbox(
                                    checked = selectedDocIds.contains(doc.id),
                                    onCheckedChange = { isChecked ->
                                        if (isChecked == true) {
                                            selectedDocIds.clear()
                                            selectedDocIds.add(doc.id)
                                            if (title.isBlank()) {
                                                title = "${doc.name.substringBeforeLast(".")} Chat"
                                            }
                                        } else {
                                            selectedDocIds.remove(doc.id)
                                        }
                                    }
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Column {
                                    Text(doc.name, fontWeight = FontWeight.Medium)
                                    Text("${doc.pageCount} pages", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        val finalTitle = title.ifBlank { "Document Chat" }
                        coroutineScope.launch {
                            val convId = viewModel.startConversation(finalTitle, selectedDocIds.toList())
                            navController.navigate("chat/$convId") {
                                popUpTo("chats") { inclusive = false }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = selectedDocIds.isNotEmpty()
                ) {
                    Icon(Icons.Default.Description, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start Document RAG Chat")
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(convId: Long, viewModel: MainViewModel, navController: NavHostController) {
    val context = LocalContext.current
    val conversations by viewModel.conversations.collectAsState()
    val chat = conversations.find { it.id == convId }
    val messages by viewModel.messagesForConversation(convId).collectAsState()
    val chatState by viewModel.chatState.collectAsState()
    val queueState by viewModel.questionQueueState.collectAsState()
    var textInput by remember { mutableStateOf("") }
    var showQueueDialog by remember { mutableStateOf(false) }
    var showCopyDialog by remember { mutableStateOf(false) }

    val linkedDocIds = chat?.documentId?.let { listOf(it) } ?: emptyList()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(chat?.title ?: "Chat", maxLines = 1)
                        Text(
                            text = if (linkedDocIds.isEmpty()) "🤖 General AI Chat (No Document)" else "📄 Document RAG Chat",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (linkedDocIds.isEmpty()) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(
                        onClick = { showQueueDialog = true }
                    ) {
                        Icon(
                            imageVector = Icons.Default.PlaylistAdd,
                            contentDescription = "Question Queue"
                        )
                    }
                    IconButton(
                        onClick = { showCopyDialog = true }
                    ) {
                        Icon(
                            imageVector = Icons.Default.Share,
                            contentDescription = "Copy / Export Chat"
                        )
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Active Question Queue Status Banner
            if (queueState.conversationId == convId) {
                if (queueState.isActive || queueState.isPaused) {
                    QuestionQueueBanner(
                        queueState = queueState,
                        onPause = { viewModel.pauseQuestionQueue() },
                        onResume = { viewModel.resumeQuestionQueue() },
                        onStop = { viewModel.stopQuestionQueue() }
                    )
                } else if (queueState.isCompleted) {
                    QuestionQueueCompletedBanner(
                        queueState = queueState,
                        onDismiss = { viewModel.clearQuestionQueue() }
                    )
                }
            }

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages) { msg ->
                    val isUser = msg.role == "user"
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
                    ) {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                            ),
                            modifier = Modifier.fillMaxWidth(0.85f)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                if (!isUser && msg.thinkingDurationMs != null && msg.thinkingDurationMs > 0) {
                                    val seconds = msg.thinkingDurationMs / 1000.0
                                    Surface(
                                        shape = RoundedCornerShape(8.dp),
                                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                        modifier = Modifier.padding(bottom = 6.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "⚡ Thought for ${String.format(java.util.Locale.US, "%.1f", seconds)}s",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.primary,
                                                fontWeight = FontWeight.SemiBold
                                            )
                                        }
                                    }
                                }
                                Text(msg.content)
                                if (!msg.sources.isNullOrBlank()) {
                                    val chunksList = remember(msg.sources) {
                                        msg.sources.split("|||").filter { it.isNotBlank() }
                                    }
                                    if (chunksList.isNotEmpty()) {
                                        var expanded by remember(msg.id) { mutableStateOf(false) }
                                        val interactionSource = remember { MutableInteractionSource() }
                                        val isPressed by interactionSource.collectIsPressedAsState()

                                        Spacer(modifier = Modifier.height(8.dp))
                                        HorizontalDivider(color = Color.LightGray.copy(alpha = 0.5f))
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clip(RoundedCornerShape(6.dp))
                                                .background(if (isPressed) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                                .clickable(
                                                    interactionSource = interactionSource,
                                                    indication = ripple(bounded = true)
                                                ) { expanded = !expanded }
                                                .padding(vertical = 6.dp, horizontal = 4.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Text(
                                                text = if (expanded) "▼ Hide Retained Chunks (${chunksList.size})" else "▶ View Extracted Context Chunks (${chunksList.size})",
                                                style = MaterialTheme.typography.labelMedium,
                                                color = MaterialTheme.colorScheme.primary,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                        AnimatedVisibility(
                                            visible = expanded,
                                            enter = expandVertically(),
                                            exit = shrinkVertically()
                                        ) {
                                            Column(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(top = 4.dp),
                                                verticalArrangement = Arrangement.spacedBy(6.dp)
                                            ) {
                                                chunksList.forEachIndexed { idx, chunk ->
                                                    Surface(
                                                        shape = RoundedCornerShape(6.dp),
                                                        color = MaterialTheme.colorScheme.background,
                                                        modifier = Modifier.fillMaxWidth()
                                                    ) {
                                                        Column(modifier = Modifier.padding(8.dp)) {
                                                            Text(
                                                                text = "Chunk #${idx + 1}",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.secondary
                                                            )
                                                            Spacer(modifier = Modifier.height(2.dp))
                                                            Text(
                                                                text = chunk,
                                                                style = MaterialTheme.typography.bodySmall
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (chatState is ChatRepository.ChatState.Thinking) {
                    val thinkingState = chatState as ChatRepository.ChatState.Thinking
                    item {
                        var elapsedSeconds by remember(thinkingState.startedAtMs) { mutableStateOf(0f) }
                        LaunchedEffect(thinkingState.startedAtMs) {
                            while (true) {
                                elapsedSeconds = (System.currentTimeMillis() - thinkingState.startedAtMs) / 1000f
                                kotlinx.coroutines.delay(100)
                            }
                        }
                        Surface(
                            shape = RoundedCornerShape(12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f),
                            modifier = Modifier
                                .fillMaxWidth(0.85f)
                                .padding(vertical = 4.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Spacer(modifier = Modifier.width(10.dp))
                                Text(
                                    text = "Thinking (${String.format(java.util.Locale.US, "%.1f", elapsedSeconds)}s)...",
                                    color = MaterialTheme.colorScheme.primary,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                    }
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = { showQueueDialog = true },
                    enabled = !queueState.isActive
                ) {
                    Icon(
                        imageVector = Icons.Default.PlaylistAdd,
                        contentDescription = "Question Queue",
                        tint = if (queueState.isActive) Color.Gray else MaterialTheme.colorScheme.primary
                    )
                }
                OutlinedTextField(
                    value = textInput,
                    onValueChange = { textInput = it },
                    placeholder = {
                        Text(if (linkedDocIds.isEmpty()) "Ask the AI model anything..." else "Ask about document...")
                    },
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                IconButton(
                    onClick = {
                        if (textInput.isNotBlank()) {
                            viewModel.sendMessage(convId, textInput, linkedDocIds)
                            textInput = ""
                        }
                    },
                    enabled = textInput.isNotBlank() && chatState !is ChatRepository.ChatState.Thinking && !queueState.isActive
                ) {
                    Icon(Icons.Default.Send, contentDescription = "Send")
                }
            }
        }
    }

    if (showQueueDialog) {
        QuestionQueueDialog(
            convId = convId,
            linkedDocIds = linkedDocIds,
            viewModel = viewModel,
            onDismiss = { showQueueDialog = false }
        )
    }

    if (showCopyDialog) {
        CopyChatDialog(
            conversationId = convId,
            conversationTitle = chat?.title ?: "Chat $convId",
            documentId = chat?.documentId,
            messages = messages,
            onDismiss = { showCopyDialog = false }
        )
    }
}

@Composable
fun QuestionQueueBanner(
    queueState: com.siddu.offlinedocai.ai.queue.QuestionQueueState,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.PlaylistPlay,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Column {
                        Text(
                            text = if (queueState.isPaused) "Queue Paused" else "Queue Processing",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Text(
                            text = "Question ${queueState.completedCount + 1} of ${queueState.totalQuestions}",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                        )
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (queueState.isPaused) {
                        IconButton(onClick = onResume) {
                            Icon(Icons.Default.PlayArrow, contentDescription = "Resume Queue", tint = MaterialTheme.colorScheme.primary)
                        }
                    } else {
                        IconButton(onClick = onPause) {
                            Icon(Icons.Default.Pause, contentDescription = "Pause Queue", tint = MaterialTheme.colorScheme.primary)
                        }
                    }
                    IconButton(onClick = onStop) {
                        Icon(Icons.Default.Stop, contentDescription = "Stop Queue", tint = Color.Red)
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { queueState.progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp))
            )

            if (!queueState.currentQuestion.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Current: \"${queueState.currentQuestion}\"",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Medium,
                    maxLines = 2,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }

            if (queueState.pendingQuestions.isNotEmpty()) {
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier
                        .clickable { expanded = !expanded }
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (expanded) "▲ Hide upcoming (${queueState.pendingQuestions.size})" else "▼ View upcoming (${queueState.pendingQuestions.size})",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                AnimatedVisibility(visible = expanded) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 4.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        queueState.pendingQuestions.forEachIndexed { idx, q ->
                            Text(
                                text = "${queueState.completedCount + 2 + idx}. $q",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.85f),
                                maxLines = 1
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun QuestionQueueCompletedBanner(
    queueState: com.siddu.offlinedocai.ai.queue.QuestionQueueState,
    onDismiss: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFE8F5E9)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = Color(0xFF2E7D32)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(
                        text = "Question Queue Completed!",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF1B5E20)
                    )
                    Text(
                        text = "All ${queueState.totalQuestions} questions processed successfully.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF2E7D32)
                    )
                }
            }
            IconButton(onClick = onDismiss) {
                Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = Color(0xFF2E7D32))
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuestionQueueDialog(
    convId: Long,
    linkedDocIds: List<Long>,
    viewModel: MainViewModel,
    onDismiss: () -> Unit
) {
    var rawText by remember { mutableStateOf("") }
    val detectedQuestions = remember(rawText) {
        viewModel.parseQuestions(rawText)
    }

    val sampleJson = """[
  "What is the main topic of this document?",
  "Summarize the key findings or recommendations.",
  "What are the core technical terms defined in the text?"
]"""

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.PlaylistAdd,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Question Queue", fontWeight = FontWeight.Bold)
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Text(
                    text = "Provide an array of questions in JSON format (or newline-separated). Questions will be asked sequentially one after another.",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray
                )
                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { rawText = sampleJson },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Sample JSON", style = MaterialTheme.typography.labelSmall)
                    }
                    OutlinedButton(
                        onClick = { rawText = "" },
                        enabled = rawText.isNotEmpty()
                    ) {
                        Text("Clear", style = MaterialTheme.typography.labelSmall)
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                OutlinedTextField(
                    value = rawText,
                    onValueChange = { rawText = it },
                    placeholder = {
                        Text(
                            text = "[\n  \"First question?\",\n  \"Second question?\"\n]",
                            style = MaterialTheme.typography.bodySmall
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(160.dp),
                    textStyle = MaterialTheme.typography.bodySmall
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Detection status card
                if (detectedQuestions.isNotEmpty()) {
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f)
                        ),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = "Detected ${detectedQuestions.size} question${if (detectedQuestions.size > 1) "s" else ""} in sequential order:",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            LazyColumn(modifier = Modifier.heightIn(max = 100.dp)) {
                                items(detectedQuestions.take(5)) { q ->
                                    val idx = detectedQuestions.indexOf(q) + 1
                                    Text(
                                        text = "$idx. $q",
                                        style = MaterialTheme.typography.bodySmall,
                                        maxLines = 2
                                    )
                                    Spacer(modifier = Modifier.height(2.dp))
                                }
                                if (detectedQuestions.size > 5) {
                                    item {
                                        Text(
                                            text = "...and ${detectedQuestions.size - 5} more questions",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = Color.Gray
                                        )
                                    }
                                }
                            }
                        }
                    }
                } else if (rawText.isNotBlank()) {
                    Text(
                        text = "No questions detected yet. Please enter a valid JSON array or lines.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (detectedQuestions.isNotEmpty()) {
                        viewModel.startQuestionQueue(convId, detectedQuestions, linkedDocIds)
                        onDismiss()
                    }
                },
                enabled = detectedQuestions.isNotEmpty()
            ) {
                Text("Start Queue (${detectedQuestions.size})")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun CopyChatDialog(
    conversationId: Long,
    conversationTitle: String,
    documentId: Long?,
    messages: List<MessageEntity>,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.ContentCopy,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text("Copy Conversation", fontWeight = FontWeight.Bold)
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = "Select what you would like to copy to your clipboard:",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                // Option 1: Everything (With retrieved context chunks)
                Card(
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            com.siddu.offlinedocai.ai.export.ChatExporter.copyChatToClipboard(
                                context = context,
                                conversationId = conversationId,
                                conversationTitle = conversationTitle,
                                documentId = documentId,
                                messages = messages,
                                includeChunks = true
                            )
                            onDismiss()
                        }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.primaryContainer,
                            modifier = Modifier.size(40.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    imageVector = Icons.Default.Description,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Copy Everything (with Chunks)",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = "Copies full conversation JSON including all user queries, AI responses, and retrieved document chunks.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                // Option 2: Q&A Only (JSON without retrieved chunks)
                Card(
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            com.siddu.offlinedocai.ai.export.ChatExporter.copyChatToClipboard(
                                context = context,
                                conversationId = conversationId,
                                conversationTitle = conversationTitle,
                                documentId = documentId,
                                messages = messages,
                                includeChunks = false
                            )
                            onDismiss()
                        }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.secondaryContainer,
                            modifier = Modifier.size(40.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    imageVector = Icons.Default.Chat,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.secondary,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Copy Q&A Only (No Chunks)",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = "Copies only questions and responses (JSON) without any retrieved document chunks.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                // Option 3: Clean Readable Text Q&A
                Card(
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            com.siddu.offlinedocai.ai.export.ChatExporter.copyPlainTextQnAToClipboard(
                                context = context,
                                conversationTitle = conversationTitle,
                                messages = messages
                            )
                            onDismiss()
                        }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.tertiaryContainer,
                            modifier = Modifier.size(40.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    imageVector = Icons.Default.FormatAlignLeft,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.tertiary,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Copy Q&A as Clean Text",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = "Copies clean formatted Markdown text (Q1, Answer, Q2...) for easy sharing.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun SettingsScreen(viewModel: MainViewModel) {
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val downloadedEmbeddingModels by viewModel.downloadedEmbeddingModels.collectAsState()
    val selectedEmbeddingModel by viewModel.selectedEmbeddingModel.collectAsState()
    val embeddingDownloadState by viewModel.embeddingDownloadState.collectAsState()
    val currentlyDownloadingEmbed by viewModel.currentlyDownloadingEmbeddingModel.collectAsState()

    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val downloadedLlmModels by viewModel.downloadedLlmModels.collectAsState()
    val selectedLlmModel by viewModel.selectedLlmModel.collectAsState()
    val llmDownloadState by viewModel.llmDownloadState.collectAsState()
    val currentlyDownloadingLlm by viewModel.currentlyDownloadingModel.collectAsState()
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp)
    ) {
        Text("Settings & Model Management", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(24.dp))

        // ==========================================
        // SECTION 1: EMBEDDING MODELS
        // ==========================================
        Text("Embedding Models (Vector Search)", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        Spacer(modifier = Modifier.height(4.dp))
        Text("Dense vector models used to index documents and retrieve precise text chunks for questions.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(12.dp))

        Text("Installed Embedding Models:", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(8.dp))

        if (downloadedEmbeddingModels.isEmpty()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text("No embedding models downloaded yet. Download one below to enable dense vector search.", modifier = Modifier.padding(16.dp), color = Color.Gray)
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                downloadedEmbeddingModels.forEach { file ->
                    val isSelected = selectedEmbeddingModel?.name == file.name
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                        ),
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { viewModel.selectEmbeddingModel(file) }
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(file.name, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = if (isSelected) "Active for Document Search" else "Tap to Select",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (isSelected) MaterialTheme.colorScheme.primary else Color.Gray,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                            IconButton(onClick = { viewModel.deleteEmbeddingModel(file) }) {
                                Icon(Icons.Default.Delete, contentDescription = "Delete Embedding Model", tint = Color.Red)
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text("Embedding Model Catalog (Download Options):", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(8.dp))

        if (embeddingDownloadState is DownloadState.Completed) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Embedding model download complete!", color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.SemiBold)
                    Button(onClick = { viewModel.resetEmbeddingDownloadState() }) {
                        Text("Dismiss")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        if (embeddingDownloadState is DownloadState.Error) {
            val error = embeddingDownloadState as DownloadState.Error
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Error: ${error.message}", color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    Button(onClick = { viewModel.resetEmbeddingDownloadState() }) {
                        Text("Retry")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            AvailableEmbeddingModels.list.forEach { model ->
                val isDownloaded = downloadedEmbeddingModels.any { it.name == model.fileName }
                val isSelected = selectedEmbeddingModel?.name == model.fileName
                val isDownloading = currentlyDownloadingEmbed == model.fileName && embeddingDownloadState is DownloadState.Progress
                val progress = if (isDownloading) embeddingDownloadState as DownloadState.Progress else null

                EmbeddingModelOptionCard(
                    model = model,
                    isDownloaded = isDownloaded,
                    isSelected = isSelected,
                    isDownloading = isDownloading,
                    downloadProgress = progress,
                    onDownload = {
                        viewModel.downloadEmbeddingModel(model.downloadUrl, model.fileName)
                    },
                    onSelect = {
                        val file = downloadedEmbeddingModels.firstOrNull { it.name == model.fileName }
                        if (file != null) viewModel.selectEmbeddingModel(file)
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(28.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(24.dp))

        // ==========================================
        // SECTION 2: SLM / LLM MODELS
        // ==========================================
        Text("Language Models (SLM / LLM)", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        Spacer(modifier = Modifier.height(4.dp))
        Text("On-device generative models for answering questions and synthesizing document insights.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(12.dp))

        Text("Installed SLM Models:", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(8.dp))

        if (downloadedLlmModels.isEmpty()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text("No SLM models downloaded yet. Download one from the catalog below.", modifier = Modifier.padding(16.dp), color = Color.Gray)
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                downloadedLlmModels.forEach { file ->
                    val isSelected = selectedLlmModel?.name == file.name
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                        ),
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { viewModel.selectLlmModel(file) }
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(file.name, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = if (isSelected) "Active for Chat" else "Tap to Select",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (isSelected) MaterialTheme.colorScheme.primary else Color.Gray,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                            IconButton(onClick = { viewModel.deleteLlmModel(file) }) {
                                Icon(Icons.Default.Delete, contentDescription = "Delete Model", tint = Color.Red)
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text("SLM Model Catalog (Download Options):", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(modifier = Modifier.height(4.dp))
        Text("Choose higher capacity models for better document comprehension & complex questions.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(12.dp))

        if (llmDownloadState is DownloadState.Completed) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Model download complete! Initialized for chat.", color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.SemiBold)
                    Button(onClick = { viewModel.resetLlmDownloadState() }) {
                        Text("Dismiss")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        if (llmDownloadState is DownloadState.Error) {
            val error = llmDownloadState as DownloadState.Error
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Error: ${error.message}", color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    Button(onClick = { viewModel.resetLlmDownloadState() }) {
                        Text("Retry")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            AvailableLlmModels.list.forEach { model ->
                val isDownloaded = downloadedLlmModels.any { it.name == model.fileName }
                val isSelected = selectedLlmModel?.name == model.fileName
                val isDownloading = currentlyDownloadingLlm == model.fileName && llmDownloadState is DownloadState.Progress
                val progress = if (isDownloading) llmDownloadState as DownloadState.Progress else null

                ModelOptionCard(
                    model = model,
                    isDownloaded = isDownloaded,
                    isSelected = isSelected,
                    isDownloading = isDownloading,
                    downloadProgress = progress,
                    onDownload = {
                        viewModel.downloadLlmModel(model.downloadUrl, model.fileName)
                    },
                    onSelect = {
                        val file = downloadedLlmModels.firstOrNull { it.name == model.fileName }
                        if (file != null) viewModel.selectLlmModel(file)
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}
