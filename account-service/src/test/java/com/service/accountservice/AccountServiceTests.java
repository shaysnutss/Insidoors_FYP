// package com.service.accountservice;

// import static org.mockito.Mockito.*;
// import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
// import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
// import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;

// import java.util.*;
// import com.service.accountservice.repository.AccountServiceRepository;
// import com.service.accountservice.model.Account;

// import org.junit.jupiter.api.Test;
// import org.junit.runner.RunWith;
// import org.springframework.beans.factory.annotation.Autowired;
// import org.springframework.boot.test.context.SpringBootTest;
// import org.springframework.boot.test.mock.mockito.MockBean;
// import org.springframework.context.annotation.Import;
// import org.springframework.test.context.junit4.SpringRunner;
// import org.springframework.test.web.servlet.MockMvc;


// @RunWith(SpringRunner.class)
// @Import(TestConfig.class)
// @SpringBootTest
// public class AccountServiceTests {
    
//     @MockBean private AccountServiceRepository accountServiceRepo;
//     @Autowired private MockMvc mockmvc;

//     @Test
//     void shouldReturnListOfAccounts() throws Exception {
//         List<Account> mockUsers = new ArrayList<>(
//             Arrays.asList(new Account(1L, "Mary", "marye@gmail.com", "ksjdusenfh", "SOC"),
//             new Account(2L, "Henry", "henry@gmail.com", "ajjr4820n", "Engineer")));
    
//         when(accountServiceRepo.findAll()).thenReturn(mockUsers);
//         mockmvc.perform(get("/api/v1/account/getAllAccounts"))
//             .andExpect(status().isOk())
//             .andExpect(jsonPath("$.size()").value(mockUsers.size()))
//             .andDo(print());
//     }

//     @Test
//     void shouldReturnAccount() throws Exception {

//         long id = 1L;
//         Account mockUser = new Account(id, "Mary", "marye@gmail.com", "ksjdusenfh", "SOC");

//         when(accountServiceRepo.findById(any())).thenReturn(Optional.of(mockUser));
//         mockmvc.perform(get("/api/v1/account/getAccountById/1"))
//             .andExpect(status().isOk())
//             .andExpect(content().contentType("application/json"))
//             .andExpect(jsonPath("$.name").value(mockUser.getName()))
//             .andExpect(jsonPath("$.password").value(mockUser.getPassword()))
//             .andExpect(jsonPath("$.email").value(mockUser.getEmail()))
//             .andExpect(jsonPath("$.role").value(mockUser.getRole()))
//             .andDo(print());
//     }

//     @Test
//     void shouldReturnAccountByEmail() throws Exception {
//         long id = 1L;
//         Account mockUser = new Account(id, "Mary", "marye@gmail.com", "ksjdusenfh", "SOC");

//         when(accountServiceRepo.findByEmail(anyString())).thenReturn(mockUser);
//         mockmvc.perform(get("/api/v1/account/getAccountByEmail/marye@gmail.com"))
//             .andExpect(status().isOk())
//             .andExpect(content().contentType("application/json"))
//             .andExpect(jsonPath("$.name").value(mockUser.getName()))
//             .andExpect(jsonPath("$.password").value(mockUser.getPassword()))
//             .andExpect(jsonPath("$.email").value(mockUser.getEmail()))
//             .andExpect(jsonPath("$.role").value(mockUser.getRole()))
//             .andDo(print());
//     }

//     @Test
//     void shouldReturnNameById() throws Exception {
//         long id = 1L;
//         Account mockUser = new Account(id, "Mary", "marye@gmail.com", "ksjdusenfh", "SOC");

//         when(accountServiceRepo.findById(any())).thenReturn(Optional.of(mockUser));
//         mockmvc.perform(get("/api/v1/account/getNameById/1"))
//             .andExpect(status().isOk())
//             .andExpect(content().string("Mary"))
//             .andDo(print());
//     }
// }
