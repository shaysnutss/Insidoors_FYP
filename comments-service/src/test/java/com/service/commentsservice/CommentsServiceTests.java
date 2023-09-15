package com.service.commentsservice;

import static org.mockito.Mockito.*;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;

import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(CommentsServiceController.class)
public class CommentsServiceTests {
    
    @MockBean private CommentsServiceRepository commentsServiceRepo;
    @Autowired private MockMvc mockMvc;
    @Autowired private ObjectMapper objectMapper;

    @Test 
    void shouldCreateComment() throws Exception {
        CommentsService mockComment = new CommentsService(1L, "first comment", 20, 14);

        mockMvc.perform(post("/api/v1/comments").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockComment)))
            .andExpect(status().isCreated())
            .andDo(print());
    }

    @Test
    void shouldReturnListOfComments() throws Exception {
        List<CommentsService> mockComments = new ArrayList<>(
            Arrays.asList(new CommentsService(1L, "first comment", 20, 14),
            new CommentsService(2L, "first comment", 44, 32)));
    
        when(commentsServiceRepo.findAll()).thenReturn(mockComments);
        mockMvc.perform(get("/api/v1/comments"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockComments.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnComment() throws Exception {
        long id = 1L;
        CommentsService mockComment = new CommentsService(id, "first comment", 20, 14);

        when(commentsServiceRepo.findById(any())).thenReturn(Optional.of(mockComment));
        mockMvc.perform(get("/api/v1/comments/1"))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.commentDescription").value(mockComment.getCommentDescription()))
            .andExpect(jsonPath("$.taskManagementId").value(mockComment.getTaskManagementId()))
            .andExpect(jsonPath("$.accountId").value(mockComment.getAccountId()))
            .andDo(print());
    }

    @Test
    void shouldReturnCommentsByTaskManagementId() throws Exception {
        long id = 1L;
        List<CommentsService> mockComments = new ArrayList<>(
            Arrays.asList(new CommentsService(id, "first comment", 20, 14)));

        when(commentsServiceRepo.findAllByTaskManagementId(anyInt())).thenReturn(mockComments);
        mockMvc.perform(get("/api/v1/comments/taskManagement/20"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockComments.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnCommentsByAccountId() throws Exception {
        long id = 1L;
        List<CommentsService> mockComments = new ArrayList<>(
            Arrays.asList(new CommentsService(id, "first comment", 20, 14)));

        when(commentsServiceRepo.findAllByAccountId(anyInt())).thenReturn(mockComments);
        mockMvc.perform(get("/api/v1/comments/account/14"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockComments.size()))
            .andDo(print());
    }

    @Test
    void shouldUpdateComment() throws Exception {
        long id = 1L;

        CommentsService mockComment = new CommentsService(id, "first comment", 20, 14);
        CommentsService mockCommentUpdate = new CommentsService(id, "second comment", 20, 14);

        when(commentsServiceRepo.findById(any())).thenReturn(Optional.of(mockComment));
        when(commentsServiceRepo.save(any(CommentsService.class))).thenReturn(mockCommentUpdate);

        mockMvc.perform(put("/api/v1/comments/1").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockCommentUpdate)))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.commentDescription").value(mockCommentUpdate.getCommentDescription()))
            .andExpect(jsonPath("$.taskManagementId").value(mockCommentUpdate.getTaskManagementId()))
            .andExpect(jsonPath("$.accountId").value(mockCommentUpdate.getAccountId()))
            .andDo(print());
    }

    @Test
    void shouldDeleteComment() throws Exception {
        long id = 1L;

        doNothing().when(commentsServiceRepo).deleteById(id);
        mockMvc.perform(delete("/api/v1/comments/1"))
            .andExpect(status().isNoContent())
            .andDo(print());
    }

}
