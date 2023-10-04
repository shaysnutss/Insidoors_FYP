package com.service.behavioralanalysisservice;

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

@WebMvcTest(BehavioralAnalysisServiceController.class)
public class BehavioralAnalysisServiceUnitTests {

    @MockBean private BehavioralAnalysisServiceRepository bAServiceRepo;
    @Autowired private MockMvc mockMvc;
    @Autowired private ObjectMapper objectMapper;
    
    @Test 
    void shouldCreateBA() throws Exception {
        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(1L, 23, 200, 12);

        mockMvc.perform(post("/api/v1/behavioralanalysis").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockBA)))
            .andExpect(status().isCreated())
            .andDo(print());
    }

    @Test
    void shouldReturnListOfBA() throws Exception {
        List<BehavioralAnalysisService> mockBAs = new ArrayList<>(
            Arrays.asList(new BehavioralAnalysisService(1L, 23, 200, 12),
            new BehavioralAnalysisService(2L, 12, 120, 34)));
    
        when(bAServiceRepo.findAll()).thenReturn(mockBAs);
        mockMvc.perform(get("/api/v1/behavioralanalysis"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.size()").value(mockBAs.size()))
            .andDo(print());
    }

    @Test
    void shouldReturnBA() throws Exception {
        long id = 1L;
        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(id, 23, 200, 12);

        when(bAServiceRepo.findById(any())).thenReturn(Optional.of(mockBA));
        mockMvc.perform(get("/api/v1/behavioralanalysis/1"))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.employeeId").value(mockBA.getEmployeeId()))
            .andExpect(jsonPath("$.riskRating").value(mockBA.getRiskRating()))
            .andExpect(jsonPath("$.suspectedCases").value(mockBA.getSuspectedCases()))
            .andDo(print());
    }

    @Test
    void shouldReturnSuspectedCasesByEmployeeId() throws Exception {
        long id = 1L;
        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(id, 23, 200, 12);

        when(bAServiceRepo.findByEmployeeId(anyInt())).thenReturn(mockBA);
        mockMvc.perform(get("/api/v1/suspectedcases/23"))
            .andExpect(status().isOk())
            .andExpect(content().string(String.valueOf(12)))
            .andDo(print());
    }

    @Test
    void shouldReturnRiskRatingByEmployeeId() throws Exception {
        long id = 1L;
        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(id, 23, 200, 12);

        when(bAServiceRepo.findByEmployeeId(anyInt())).thenReturn(mockBA);
        mockMvc.perform(get("/api/v1/riskrating/23"))
            .andExpect(status().isOk())
            .andExpect(content().string(String.valueOf(200)))
            .andDo(print());
    }

    @Test
    void shouldUpdateRiskRating() throws Exception {
        long id = 1L;

        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(id, 23, 200, 12);
        BehavioralAnalysisService mockBAUpdate = new BehavioralAnalysisService(id, 23, 170, 12);

        when(bAServiceRepo.findByEmployeeId(anyInt())).thenReturn(mockBA);
        when(bAServiceRepo.save(any(BehavioralAnalysisService.class))).thenReturn(mockBAUpdate);

        mockMvc.perform(put("/api/v1/updateRiskRating/23").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockBAUpdate)))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.employeeId").value(mockBAUpdate.getEmployeeId()))
            .andExpect(jsonPath("$.riskRating").value(370))
            .andExpect(jsonPath("$.suspectedCases").value(mockBAUpdate.getSuspectedCases()))
            .andDo(print());
    }

    @Test
    void shouldUpdateSuspectedCases() throws Exception {
        long id = 1L;

        BehavioralAnalysisService mockBA = new BehavioralAnalysisService(id, 23, 200, 12);
        BehavioralAnalysisService mockBAUpdate = new BehavioralAnalysisService(id, 23, 200, 8);

        when(bAServiceRepo.findByEmployeeId(anyInt())).thenReturn(mockBA);
        when(bAServiceRepo.save(any(BehavioralAnalysisService.class))).thenReturn(mockBAUpdate);

        mockMvc.perform(put("/api/v1/updateSuspectedCases/23").contentType(MediaType.APPLICATION_JSON)
            .content(objectMapper.writeValueAsString(mockBAUpdate)))
            .andExpect(status().isOk())
            .andExpect(content().contentType("application/json"))
            .andExpect(jsonPath("$.employeeId").value(mockBAUpdate.getEmployeeId()))
            .andExpect(jsonPath("$.riskRating").value(mockBAUpdate.getRiskRating()))
            .andExpect(jsonPath("$.suspectedCases").value(mockBAUpdate.getSuspectedCases()))
            .andDo(print());
    }

    @Test
    void shouldDeleteBA() throws Exception {
        long id = 1L;

        doNothing().when(bAServiceRepo).deleteById(id);
        mockMvc.perform(delete("/api/v1/behavioralanalysis/1"))
            .andExpect(status().isNoContent())
            .andDo(print());
    }

}
