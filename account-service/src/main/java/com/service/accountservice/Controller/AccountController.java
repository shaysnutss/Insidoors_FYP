package com.service.accountservice.Controller;

import com.service.accountservice.exception.AccountNotFoundException;
import com.service.accountservice.model.Account;
import com.service.accountservice.repository.AccountServiceRepository;
import lombok.RequiredArgsConstructor;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/account")
@RequiredArgsConstructor
public class AccountController {

    private final AccountServiceRepository accountServiceRepository;

    /**
     * Find all accounts
     *
     * @return list of all accounts registered in the database
     */
    @GetMapping("/getAllAccounts")
    public List<Account> getAllAccounts() {
        List<Account> allAccounts = accountServiceRepository.findAll();
        if (allAccounts == null){
            throw new AccountNotFoundException();
        }
        //remove password from all accounts before sending
        List<Account> allAccountsNoPassword = new ArrayList<>();
        for (Account account : allAccounts) {
            allAccountsNoPassword.add(Account.accountNoPassword(account));
        }
        return allAccountsNoPassword;
    }


    /**
     * Get a specific account by ID
     *
     * @param id account ID
     * @return specific account associated with that ID
     */
    @GetMapping("/getAccountById/{id}")
    public Account getAccountById(@PathVariable(value = "id") long id) {
        //long numericValue = Long.parseLong(id);
        Optional<Account> account = accountServiceRepository.findById(id);
        if (account.isEmpty()) {
            throw new AccountNotFoundException(id);
        }
        return Account.accountNoPassword(account.get());
    }


    //get account by email
    /**
     * Get a specific account by email
     *
     * @param email account email
     * @return specific account associated with that ID
     */
    @GetMapping("/getAccountByEmail/{email}")
    public Account getAccountByEmail(@PathVariable(value = "email") String email) {
        Account account = accountServiceRepository.findByEmail(email);
        if (account == null) {
            throw new AccountNotFoundException(email);
        }
        return Account.accountNoPassword(account);
    }


    //get name by id
    /**
     * Get a name of a specific account by ID
     *
     * @param id account ID
     * @return name of specific account associated with that ID
     */
    @GetMapping("/getNameById/{id}")
    public String getNameById(@PathVariable(value = "id") long id) {
        Optional<Account> account = accountServiceRepository.findById(id);
        if (account.isEmpty()) {
            throw new AccountNotFoundException(id);
        }
        return account.get().getName();
    }

}
