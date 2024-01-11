echo "Running experiments for DecByzPG"

for ENV in CartPole LunarLander; do
    if [[ "$ENV" == "CartPole" ]]
    then
        LR="5e-4"
    else
        LR="1e-3"
    fi

    for S in 0 1 2 3 4 5 6 7 8 9; do
        python run.py --lr $LR --env $ENV --num_workers 1 --num_byz 0 --agreement_type none --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 5 --num_byz 0 --agreement_type none --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 0 --agreement_type none --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 0 --agreement_type mda --attack_type none --aggregation_type rfa --seed $S
    done

    for ATCK in random_action random_unif_-1000_1000 avg_zero; do
        for S in 0 1 2 3 4 5 6 7 8 9; do
            python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 3 --agreement_type none --attack_type $ATCK --aggregation_type avg --seed $S
            python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 3 --agreement_type mda --attack_type $ATCK --aggregation_type rfa --seed $S
        done
    done
done


echo "Running experiments for ByzPG"

for ENV in CartPole LunarLander; do
    if [[ "$ENV" == "CartPole" ]]
    then
        LR="5e-4"
    else
        LR="1e-3"
    fi

    for S in 0 1 2 3 4 5 6 7 8 9; do
        python run.py --lr $LR --env $ENV --num_workers 1 --num_byz 0 --agreement_type centralized --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 5 --num_byz 0 --agreement_type centralized --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 0 --agreement_type centralized --attack_type none --aggregation_type avg --seed $S
        python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 0 --agreement_type centralized --attack_type none --aggregation_type rfa --seed $S
    done

    for ATCK in random_action random_unif_-1000_1000 avg_zero; do
        for S in 0 1 2 3 4 5 6 7 8 9; do
            python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 3 --agreement_type centralized --attack_type $ATCK --aggregation_type avg --seed $S
            python run.py --lr $LR --env $ENV --num_workers 13 --num_byz 3 --agreement_type centralized --attack_type $ATCK --aggregation_type rfa --seed $S
        done
    done
done